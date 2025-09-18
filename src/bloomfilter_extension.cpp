// bloomfilter_extension.cpp — updated for DuckDB v1.4+
//
// Requires: src/include/bloomfilter_extension.hpp declares
//   void Load(duckdb::ExtensionLoader &loader) override;

#include "bloomfilter_extension.hpp"
#include "duckdb.hpp"

// core includes from template
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension.hpp" // ExtensionLoader, DatabaseInstance

// (leftover from template; harmless even if not used)
#include <openssl/opensslv.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace duckdb {

// ============================
// Small helpers (no deps)
// ============================

static inline void SetBit(data_ptr_t bytes, uint64_t idx) {
	const uint64_t byte = idx >> 3;          // idx / 8
	const uint8_t  mask = 1u << (idx & 7u);  // idx % 8
	bytes[byte] |= mask;
}

static inline bool TestBit(const_data_ptr_t bytes, uint64_t idx) {
	const uint64_t byte = idx >> 3;
	const uint8_t  mask = 1u << (idx & 7u);
	return (bytes[byte] & mask) != 0;
}

// very small 64-bit FNV-1a; we salt w/ seed for variability
static inline uint64_t FNV1a64(const void *data, size_t len, uint64_t seed) {
	const uint8_t *p = static_cast<const uint8_t *>(data);
	uint64_t h = 14695981039346656037ULL ^ seed;
	const uint64_t prime = 1099511628211ULL;
	for (size_t i = 0; i < len; i++) { h ^= p[i]; h *= prime; }
	return h;
}

static inline void HashFromValue(const Value &v, uint64_t seed, uint64_t &h1, uint64_t &h2) {
	switch (v.type().id()) {
	case LogicalTypeId::TINYINT:    { auto x = v.GetValue<int8_t>();    h1 = FNV1a64(&x,sizeof(x),seed); h2 = FNV1a64(&x,sizeof(x),seed^0x9E3779B97F4A7C15ULL); return; }
	case LogicalTypeId::SMALLINT:   { auto x = v.GetValue<int16_t>();   h1 = FNV1a64(&x,sizeof(x),seed); h2 = FNV1a64(&x,sizeof(x),seed^0x9E3779B97F4A7C15ULL); return; }
	case LogicalTypeId::INTEGER:    { auto x = v.GetValue<int32_t>();   h1 = FNV1a64(&x,sizeof(x),seed); h2 = FNV1a64(&x,sizeof(x),seed^0x9E3779B97F4A7C15ULL); return; }
	case LogicalTypeId::BIGINT:     { auto x = v.GetValue<int64_t>();   h1 = FNV1a64(&x,sizeof(x),seed); h2 = FNV1a64(&x,sizeof(x),seed^0x9E3779B97F4A7C15ULL); return; }
	case LogicalTypeId::HUGEINT:    { auto x = v.GetValue<hugeint_t>(); h1 = FNV1a64(&x,sizeof(x),seed); h2 = FNV1a64(&x,sizeof(x),seed^0x9E3779B97F4A7C15ULL); return; }
	case LogicalTypeId::FLOAT:      { auto x = v.GetValue<float>();     if (x==0.0f) x = 0.0f; h1 = FNV1a64(&x,sizeof(x),seed); h2 = FNV1a64(&x,sizeof(x),seed^0x9E3779B97F4A7C15ULL); return; }
	case LogicalTypeId::DOUBLE:     { auto x = v.GetValue<double>();    if (x==0.0)  x = 0.0;  h1 = FNV1a64(&x,sizeof(x),seed); h2 = FNV1a64(&x,sizeof(x),seed^0x9E3779B97F4A7C15ULL); return; }
	case LogicalTypeId::BLOB:
	case LogicalTypeId::VARCHAR:    { auto s = StringValue::Get(v);     h1 = FNV1a64(s.data(), s.size(), seed); h2 = FNV1a64(s.data(), s.size(), seed^0x9E3779B97F4A7C15ULL); return; }
	default:                        { auto s = v.ToString();            h1 = FNV1a64(s.data(), s.size(), seed); h2 = FNV1a64(s.data(), s.size(), seed^0x9E3779B97F4A7C15ULL); return; }
	}
}

// ============================
// bloom_calc_m / bloom_calc_k
// ============================

static void BloomCalcM(DataChunk &args, ExpressionState &, Vector &result) {
	auto ndv = FlatVector::GetData<int64_t>(args.data[0]);
	auto fpp = FlatVector::GetData<double>(args.data[1]);
	auto out = FlatVector::GetData<int32_t>(result);

	for (idx_t i = 0; i < args.size(); i++) {
		if (FlatVector::IsNull(args.data[0], i) || FlatVector::IsNull(args.data[1], i)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		double n = (double)std::max<int64_t>(1, ndv[i]);
		double p = std::min(std::max(fpp[i], 1e-12), 0.5); // clamp
		const double ln2 = 0.69314718055994530941723212145818;
		double m_d = std::ceil((-n * std::log(p)) / (ln2*ln2));
		int64_t m64 = (int64_t)std::max(8.0, m_d);
		if (m64 > (int64_t)INT32_MAX) m64 = INT32_MAX;
		out[i] = (int32_t)m64;
	}
}

static void BloomCalcK(DataChunk &args, ExpressionState &, Vector &result) {
	auto m    = FlatVector::GetData<int32_t>(args.data[0]);
	auto ndv  = FlatVector::GetData<int64_t>(args.data[1]);
	auto fpp  = FlatVector::GetData<double>(args.data[2]);
	(void)fpp; // API includes fpp; formula uses m,ndv
	auto out  = FlatVector::GetData<int32_t>(result);

	for (idx_t i = 0; i < args.size(); i++) {
		if (FlatVector::IsNull(args.data[0], i) || FlatVector::IsNull(args.data[1], i) || FlatVector::IsNull(args.data[2], i)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		double n   = (double)std::max<int64_t>(1, ndv[i]);
		double m_d = (double)std::max<int32_t>(8, m[i]);
		const double ln2 = 0.69314718055994530941723212145818;
		int64_t k64 = (int64_t)std::llround((m_d / n) * ln2);
		if (k64 < 1)  k64 = 1;
		if (k64 > 32) k64 = 32; // practical cap
		out[i] = (int32_t)k64;
	}
}

// ==============================================
// bloom_maybe_contains(bitset, m, k, seed, value)
// ==============================================
static void BloomMaybeContains(DataChunk &args, ExpressionState &, Vector &result) {
	auto &bitset_vec = args.data[0];
	auto &m_vec      = args.data[1];
	auto &k_vec      = args.data[2];
	auto &seed_vec   = args.data[3];
	auto &val_vec    = args.data[4];

	UnifiedVectorFormat bitset_uf, m_uf, k_uf, seed_uf, val_uf;
	bitset_vec.ToUnifiedFormat(args.size(), bitset_uf);
	m_vec.ToUnifiedFormat(args.size(), m_uf);
	k_vec.ToUnifiedFormat(args.size(), k_uf);
	seed_vec.ToUnifiedFormat(args.size(), seed_uf);
	val_vec.ToUnifiedFormat(args.size(), val_uf);

	auto res = FlatVector::GetData<bool>(result);

	for (idx_t i = 0; i < args.size(); i++) {
		const idx_t bi = bitset_uf.sel->get_index(i);
		const idx_t mi = m_uf.sel->get_index(i);
		const idx_t ki = k_uf.sel->get_index(i);
		const idx_t si = seed_uf.sel->get_index(i);
		const idx_t vi = val_uf.sel->get_index(i);

		if (!bitset_uf.validity.RowIsValid(bi) || !val_uf.validity.RowIsValid(vi)) {
			res[i] = false;
			continue;
		}

		const int32_t m    = ((const int32_t*)m_uf.data)[mi];
		const int32_t k    = ((const int32_t*)k_uf.data)[ki];
		const int64_t seed = ((const int64_t*)seed_uf.data)[si];

		const string_t blob = ((const string_t*)bitset_uf.data)[bi];
		const auto *bytes   = (const_data_ptr_t)blob.GetDataUnsafe();

		const Value v = val_vec.GetValue(i);
		uint64_t h1, h2;
		HashFromValue(v, (uint64_t)seed, h1, h2);

		bool maybe = true;
		const uint64_t M = (uint64_t)m;
		for (int32_t j = 0; j < k; j++) {
			const uint64_t idx = (h1 + (uint64_t)j * h2) % M;
			if (!TestBit(bytes, idx)) { maybe = false; break; }
		}
		res[i] = maybe;
	}
}

// ============================
// bloom_build_agg (bind-less)
// ============================

struct BloomAggState {
	data_ptr_t bytes = nullptr;  // heap buffer of length bytes_len
	idx_t      bytes_len = 0;
	bool       initialized = false;

	// parameters captured from first row
	bool       params_set = false;
	int32_t    m = 0;
	int32_t    k = 0;
	int64_t    seed = 0;
};

static void BloomAggInit(const AggregateFunction &, data_ptr_t state_p) {
	auto &st = *reinterpret_cast<BloomAggState*>(state_p);
	st.bytes = nullptr;
	st.bytes_len = 0;
	st.initialized = false;
	st.params_set = false;
	st.m = 0; st.k = 0; st.seed = 0;
}

static void BloomAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	auto states = FlatVector::GetData<data_ptr_t>(state_vector);
	for (idx_t i = 0; i < count; i++) {
		auto *st = reinterpret_cast<BloomAggState*>(states[i]);
		if (st && st->bytes) {
			free(st->bytes);
			st->bytes = nullptr;
			st->bytes_len = 0;
			st->initialized = false;
			st->params_set = false;
		}
	}
}

static void EnsureStateBuffer(BloomAggState &st) {
	if (!st.initialized) {
		st.bytes_len = (idx_t)((std::max(8, st.m) + 7) / 8);
		st.bytes = (data_ptr_t)malloc(st.bytes_len);
		if (!st.bytes) throw InternalException("bloom_build_agg: OOM allocating state buffer");
		memset(st.bytes, 0, st.bytes_len);
		st.initialized = true;
	}
}

// update(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &states, idx_t count)
static void BloomAggUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector, idx_t count) {
	D_ASSERT(input_count == 4); // value, m, k, seed
	auto &val_vec  = inputs[0];
	auto &m_vec    = inputs[1];
	auto &k_vec    = inputs[2];
	auto &seed_vec = inputs[3];

	UnifiedVectorFormat vuf, muf, kuf, suf;
	val_vec.ToUnifiedFormat(count, vuf);
	m_vec.ToUnifiedFormat(count, muf);
	k_vec.ToUnifiedFormat(count, kuf);
	seed_vec.ToUnifiedFormat(count, suf);

	auto states = FlatVector::GetData<data_ptr_t>(state_vector);

	for (idx_t i = 0; i < count; i++) {
		auto &st = *reinterpret_cast<BloomAggState*>(states[i]);

		const idx_t mi = muf.sel->get_index(i);
		const idx_t ki = kuf.sel->get_index(i);
		const idx_t si = suf.sel->get_index(i);

		if (!st.params_set) {
			if (!muf.validity.RowIsValid(mi) || !kuf.validity.RowIsValid(ki) || !suf.validity.RowIsValid(si)) {
				throw BinderException("bloom_build_agg: m/k/seed must be non-NULL");
			}
			st.m    = ((const int32_t*)muf.data)[mi];
			st.k    = ((const int32_t*)kuf.data)[ki];
			st.seed = ((const int64_t*)suf.data)[si];

			if (st.m < 8) st.m = 8;
			if (st.k < 1) st.k = 1;
			if (st.k > 32) st.k = 32;

			st.params_set = true;
			EnsureStateBuffer(st);
		} else {
			// enforce same parameters throughout the group
			int32_t mm = ((const int32_t*)muf.data)[mi];
			int32_t kk = ((const int32_t*)kuf.data)[ki];
			int64_t ss = ((const int64_t*)suf.data)[si];
			if (muf.validity.RowIsValid(mi) && (mm != st.m) ) throw BinderException("bloom_build_agg: m must be constant within group");
			if (kuf.validity.RowIsValid(ki) && (kk != st.k) ) throw BinderException("bloom_build_agg: k must be constant within group");
			if (suf.validity.RowIsValid(si) && (ss != st.seed)) throw BinderException("bloom_build_agg: seed must be constant within group");
		}

		const idx_t vi = vuf.sel->get_index(i);
		if (!vuf.validity.RowIsValid(vi)) continue; // skip NULL value

		// hash the value
		Value v = val_vec.GetValue(i);
		uint64_t h1, h2;
		HashFromValue(v, (uint64_t)st.seed, h1, h2);

		const uint64_t M = (uint64_t)st.m;
		for (int32_t j = 0; j < st.k; j++) {
			const uint64_t idx = (h1 + (uint64_t)j * h2) % M;
			SetBit(st.bytes, idx);
		}
	}
}

// combine(Vector &source, Vector &target, AggregateInputData &, idx_t count)
static void BloomAggCombine(Vector &source, Vector &target, AggregateInputData &, idx_t count) {
	auto src_states = FlatVector::GetData<data_ptr_t>(source);
	auto dst_states = FlatVector::GetData<data_ptr_t>(target);

	for (idx_t i = 0; i < count; i++) {
		auto &src = *reinterpret_cast<BloomAggState*>(src_states[i]);
		auto &dst = *reinterpret_cast<BloomAggState*>(dst_states[i]);

		if (!src.initialized) continue;

		if (!dst.params_set) {
			// adopt src params
			dst.m = src.m; dst.k = src.k; dst.seed = src.seed; dst.params_set = true;
		} else {
			if (dst.m != src.m || dst.k != src.k || dst.seed != src.seed) {
				throw BinderException("bloom_build_agg: combining states with different m/k/seed");
			}
		}
		if (!dst.initialized) {
			dst.bytes_len = src.bytes_len;
			dst.bytes = (data_ptr_t)malloc(dst.bytes_len);
			if (!dst.bytes) throw InternalException("bloom_build_agg: OOM allocating combine buffer");
			memset(dst.bytes, 0, dst.bytes_len);
			dst.initialized = true;
		}
		for (idx_t b = 0; b < src.bytes_len; b++) {
			dst.bytes[b] |= src.bytes[b];
		}
	}
}

// finalize(Vector &states, AggregateInputData &, Vector &result, idx_t count, idx_t /*offset*/)
static void BloomAggFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t /*offset*/) {
	auto states = FlatVector::GetData<data_ptr_t>(state_vector);
	auto out = FlatVector::GetData<string_t>(result);

	for (idx_t i = 0; i < count; i++) {
		auto *st = reinterpret_cast<BloomAggState*>(states[i]);
		if (!st || !st->initialized) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		// hand back a blob of size bytes_len
		out[i] = StringVector::AddString(result, (const char *)st->bytes, st->bytes_len);
	}
}

// =======================
// (template demo fns)
// =======================
inline void BloomfilterScalarFun(DataChunk &args, ExpressionState &, Vector &result) {
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		return StringVector::AddString(result, "Bloomfilter " + name.GetString() + " 🐥");
	});
}

inline void BloomfilterOpenSSLVersionScalarFun(DataChunk &args, ExpressionState &, Vector &result) {
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		return StringVector::AddString(result, "Bloomfilter " + name.GetString() + ", my linked OpenSSL version is " + OPENSSL_VERSION_TEXT);
	});
}

// =======================
// Registration (via ExtensionLoader)
// =======================
static void LoadInternal(ExtensionLoader &loader) {
	// calc m/k
	loader.RegisterFunction(ScalarFunction(
		"bloom_bits_required",
		{LogicalType::BIGINT, LogicalType::DOUBLE},
		LogicalType::INTEGER,
		BloomCalcM));

	loader.RegisterFunction(ScalarFunction(
		"bloom_optimal_hashes",
		{LogicalType::INTEGER, LogicalType::BIGINT, LogicalType::DOUBLE},
		LogicalType::INTEGER,
		BloomCalcK));

	// probe
	loader.RegisterFunction(ScalarFunction(
		"bloom_might_contain",
		{LogicalType::BLOB, LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::BIGINT, LogicalType::ANY},
		LogicalType::BOOLEAN,
		BloomMaybeContains));

	// aggregate: value:any, m:int, k:int, seed:bigint -> blob (no bind)
	AggregateFunction bloom_build(
		"bloom_build_bytes_agg",
		{ LogicalType::ANY, LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::BIGINT },
		LogicalType::BLOB,
		/* state_size */ [](const AggregateFunction &) -> idx_t { return sizeof(BloomAggState); },
		/* init       */  BloomAggInit,
		/* update     */  BloomAggUpdate,
		/* combine    */  BloomAggCombine,
		/* finalize   */  BloomAggFinalize,
		/* null_handling */ FunctionNullHandling::DEFAULT_NULL_HANDLING
	);
	bloom_build.destructor = BloomAggDestroy;
	loader.RegisterFunction(bloom_build);

	// template demos
	loader.RegisterFunction(ScalarFunction("bloomfilter",
		{LogicalType::VARCHAR}, LogicalType::VARCHAR, BloomfilterScalarFun));
	loader.RegisterFunction(ScalarFunction("bloomfilter_openssl_version",
		{LogicalType::VARCHAR}, LogicalType::VARCHAR, BloomfilterOpenSSLVersionScalarFun));
}

void BloomfilterExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string BloomfilterExtension::Name() { return "bloomfilter"; }

std::string BloomfilterExtension::Version() const {
#ifdef EXT_VERSION_BLOOMFILTER
	return EXT_VERSION_BLOOMFILTER;
#else
	return "";
#endif
}

} // namespace duckdb

// ---- Loadable extension entrypoint for DuckDB >= 1.4 ----
// (DuckDB's build system expects this exact symbol name)
extern "C" {

DUCKDB_EXTENSION_API void bloomfilter_duckdb_cpp_init(duckdb::ExtensionLoader &loader) {
	duckdb::LoadInternal(loader);
}

} // extern "C"
