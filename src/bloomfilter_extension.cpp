#define DUCKDB_EXTENSION_MAIN

#include "bloomfilter_extension.hpp"
#include "duckdb.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

// Template leftover; harmless if present
#include <openssl/opensslv.h>

#include <cmath>
#include <cstring>
#include <algorithm>

namespace duckdb {

// ================================
// Helpers (no external dependencies)
// ================================

// Work with raw byte pointers to avoid typedef mismatches
static inline void SetBit(unsigned char *bytes, uint64_t idx) {
	const uint64_t byte = idx >> 3;          // idx / 8
	const unsigned char  mask = 1u << (idx & 7u);  // idx % 8
	bytes[byte] |= mask;
}
static inline bool TestBit(const unsigned char *bytes, uint64_t idx) {
	const uint64_t byte = idx >> 3;
	const unsigned char  mask = 1u << (idx & 7u);
	return (bytes[byte] & mask) != 0;
}

// Simple 64-bit FNV-1a; salted with 'seed'
static inline uint64_t FNV1a64(const void *data, size_t len, uint64_t seed) {
	const unsigned char *p = static_cast<const unsigned char *>(data);
	uint64_t h = 14695981039346656037ULL ^ seed;         // offset basis ^ seed
	const uint64_t prime = 1099511628211ULL;
	for (size_t i = 0; i < len; i++) {
		h ^= p[i];
		h *= prime;
	}
	return h;
}

// Convert arbitrary Value to two 64-bit hashes (double hashing h1,h2)
static inline void HashFromValue(const Value &v, uint64_t seed, uint64_t &h1, uint64_t &h2) {
	switch (v.type().id()) {
	case LogicalTypeId::TINYINT: { int8_t  x = v.GetValue<int8_t>();  h1 = FNV1a64(&x,sizeof(x),seed); h2 = FNV1a64(&x,sizeof(x),seed^0x9E3779B97F4A7C15ULL); return; }
	case LogicalTypeId::SMALLINT:{ int16_t x = v.GetValue<int16_t>(); h1 = FNV1a64(&x,sizeof(x),seed); h2 = FNV1a64(&x,sizeof(x),seed^0x9E3779B97F4A7C15ULL); return; }
	case LogicalTypeId::INTEGER: { int32_t x = v.GetValue<int32_t>(); h1 = FNV1a64(&x,sizeof(x),seed); h2 = FNV1a64(&x,sizeof(x),seed^0x9E3779B97F4A7C15ULL); return; }
	case LogicalTypeId::BIGINT:  { int64_t x = v.GetValue<int64_t>(); h1 = FNV1a64(&x,sizeof(x),seed); h2 = FNV1a64(&x,sizeof(x),seed^0x9E3779B97F4A7C15ULL); return; }
	case LogicalTypeId::HUGEINT: { hugeint_t x = v.GetValue<hugeint_t>(); h1 = FNV1a64(&x,sizeof(x),seed); h2 = FNV1a64(&x,sizeof(x),seed^0x9E3779B97F4A7C15ULL); return; }
	case LogicalTypeId::FLOAT:   { float  x = v.GetValue<float>();  if (x==0.0f) x=0.0f;  h1 = FNV1a64(&x,sizeof(x),seed); h2 = FNV1a64(&x,sizeof(x),seed^0x9E3779B97F4A7C15ULL); return; }
	case LogicalTypeId::DOUBLE:  { double x = v.GetValue<double>(); if (x==0.0)  x=0.0;   h1 = FNV1a64(&x,sizeof(x),seed); h2 = FNV1a64(&x,sizeof(x),seed^0x9E3779B97F4A7C15ULL); return; }
	case LogicalTypeId::BLOB:
	case LogicalTypeId::VARCHAR: { auto s = StringValue::Get(v); h1 = FNV1a64(s.data(), s.size(), seed); h2 = FNV1a64(s.data(), s.size(), seed^0x9E3779B97F4A7C15ULL); return; }
	default: { auto s = v.ToString(); h1 = FNV1a64(s.data(), s.size(), seed); h2 = FNV1a64(s.data(), s.size(), seed^0x9E3779B97F4A7C15ULL); return; }
	}
}

// ================================
//  bloom_calc_m / bloom_calc_k
// ================================

static inline int32_t BloomCalcMInternal(int64_t ndv, double fpp) {
	if (ndv <= 0) return 8; // minimal non-zero filter
	if (!(fpp > 0.0 && fpp < 1.0)) {
		throw InvalidInputException("bloom_calc_m: fpp must be in (0,1)");
	}
	const double ln2 = std::log(2.0);
	const double m   = - (double)ndv * std::log(fpp) / (ln2 * ln2);
	const int64_t mi = (int64_t)std::ceil(m);
	return (int32_t)std::max<int64_t>(8, mi);
}

static inline int32_t BloomCalcKInternal(int32_t m, int64_t ndv /*, double fpp_unused*/) {
	if (m <= 0) throw InvalidInputException("bloom_calc_k: m must be > 0");
	if (ndv <= 0) return 1;
	const double ln2 = std::log(2.0);
	const double k   = ((double)m / (double)ndv) * ln2;
	return (int32_t)std::max<int>(1, (int)std::round(k));
}

static void BloomCalcMScalar(DataChunk &args, ExpressionState &, Vector &result) {
	auto &ndv_vec = args.data[0];  // BIGINT
	auto &fpp_vec = args.data[1];  // DOUBLE

	UnifiedVectorFormat ndv_uf, fpp_uf;
	ndv_vec.ToUnifiedFormat(args.size(), ndv_uf);
	fpp_vec.ToUnifiedFormat(args.size(), fpp_uf);

	auto out = FlatVector::GetData<int32_t>(result);
	for (idx_t i = 0; i < args.size(); i++) {
		const idx_t ni = ndv_uf.sel->get_index(i);
		const idx_t fi = fpp_uf.sel->get_index(i);
		if (!ndv_uf.validity.RowIsValid(ni) || !fpp_uf.validity.RowIsValid(fi)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		const int64_t ndv = ((const int64_t*)ndv_uf.data)[ni];
		const double  fpp = ((const double *)fpp_uf.data)[fi];
		out[i] = BloomCalcMInternal(ndv, fpp);
	}
}

static void BloomCalcKScalar(DataChunk &args, ExpressionState &, Vector &result) {
	auto &m_vec   = args.data[0];   // INTEGER
	auto &ndv_vec = args.data[1];   // BIGINT
	auto &fpp_vec = args.data[2];   // DOUBLE (included per API)
	(void)fpp_vec;

	UnifiedVectorFormat m_uf, ndv_uf, fpp_uf;
	m_vec.ToUnifiedFormat(args.size(), m_uf);
	ndv_vec.ToUnifiedFormat(args.size(), ndv_uf);
	fpp_vec.ToUnifiedFormat(args.size(), fpp_uf);

	auto out = FlatVector::GetData<int32_t>(result);
	for (idx_t i = 0; i < args.size(); i++) {
		const idx_t mi = m_uf.sel->get_index(i);
		const idx_t ni = ndv_uf.sel->get_index(i);
		const idx_t fi = fpp_uf.sel->get_index(i);
		if (!m_uf.validity.RowIsValid(mi) || !ndv_uf.validity.RowIsValid(ni) || !fpp_uf.validity.RowIsValid(fi)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		const int32_t m   = ((const int32_t*)m_uf.data)[mi];
		const int64_t ndv = ((const int64_t*)ndv_uf.data)[ni];
		out[i] = BloomCalcKInternal(m, ndv);
	}
}

// ==============================================
// bloom_build(values LIST<ANY>, m, k, seed) -> BLOB
// (uses Value API only; no list headers)
// ==============================================
static void BloomBuildScalar(DataChunk &args, ExpressionState &, Vector &result) {
	auto &list_vec = args.data[0];                   // LIST<ANY>
	auto &m_vec    = args.data[1];                   // INTEGER
	auto &k_vec    = args.data[2];                   // INTEGER
	auto &seed_vec = args.data[3];                   // BIGINT

	UnifiedVectorFormat list_uf, m_uf, k_uf, seed_uf;
	list_vec.ToUnifiedFormat(args.size(), list_uf);
	m_vec.ToUnifiedFormat(args.size(), m_uf);
	k_vec.ToUnifiedFormat(args.size(), k_uf);
	seed_vec.ToUnifiedFormat(args.size(), seed_uf);

	for (idx_t i = 0; i < args.size(); i++) {
		const idx_t li = list_uf.sel->get_index(i);
		const idx_t mi = m_uf.sel->get_index(i);
		const idx_t ki = k_uf.sel->get_index(i);
		const idx_t si = seed_uf.sel->get_index(i);

		if (!list_uf.validity.RowIsValid(li) || !m_uf.validity.RowIsValid(mi) ||
		    !k_uf.validity.RowIsValid(ki)    || !seed_uf.validity.RowIsValid(si)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}

		const int32_t m    = ((const int32_t*)m_uf.data)[mi];
		const int32_t k    = ((const int32_t*)k_uf.data)[ki];
		const int64_t seed = ((const int64_t*)seed_uf.data)[si];

		if (m <= 0 || k <= 0) {
			throw InvalidInputException("bloom_build: m and k must be > 0");
		}

		// Get the entire LIST value as a Value and iterate its children
		Value list_value = list_vec.GetValue(li);
		if (list_value.IsNull()) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		const auto &elems = ListValue::GetChildren(list_value);

		const idx_t nbytes  = (m + 7) >> 3;
		auto out = StringVector::EmptyString(result, nbytes);
		std::memset(out.GetDataWriteable(), 0, nbytes);
		auto *out_bytes = reinterpret_cast<unsigned char *>(out.GetDataWriteable());

		for (const auto &elem : elems) {
			if (elem.IsNull()) continue;
			uint64_t h1, h2;
			HashFromValue(elem, (uint64_t)seed, h1, h2);

			const uint64_t M = (uint64_t)m;
			for (int32_t t = 0; t < k; t++) {
				const uint64_t idx = (h1 + (uint64_t)t * h2) % M;
				SetBit(out_bytes, idx);
			}
		}

		out.Finalize();
		FlatVector::GetData<string_t>(result)[i] = out;
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

		if (!bitset_uf.validity.RowIsValid(bi) || !val_uf.validity.RowIsValid(vi) ||
		    !m_uf.validity.RowIsValid(mi)      || !k_uf.validity.RowIsValid(ki) ||
		    !seed_uf.validity.RowIsValid(si)) {
			res[i] = false;
			continue;
		}

		const int32_t m    = ((const int32_t*)m_uf.data)[mi];
		const int32_t k    = ((const int32_t*)k_uf.data)[ki];
		const int64_t seed = ((const int64_t*)seed_uf.data)[si];

		const string_t blob = ((const string_t*)bitset_uf.data)[bi];
		const auto *bytes   = reinterpret_cast<const unsigned char *>(blob.GetDataUnsafe());

		const uint64_t needed = (uint64_t)((m + 7) >> 3);
		if ((uint64_t)blob.GetSize() < needed) {
			throw InvalidInputException("bloom_maybe_contains: bitset too small for m=%d (need %llu bytes, got %lld)",
			                            m, (unsigned long long)needed, (long long)blob.GetSize());
		}

		const Value v = val_vec.GetValue(vi);
		uint64_t h1, h2; HashFromValue(v, (uint64_t)seed, h1, h2);

		bool maybe = true;
		const uint64_t M = (uint64_t)m;
		for (int32_t j = 0; j < k; j++) {
			const uint64_t idx = (h1 + (uint64_t)j * h2) % M;
			if (!TestBit(bytes, idx)) { maybe = false; break; }
		}
		res[i] = maybe;
	}
}

// ==============
// Template demo fns (left in, harmless)
// ==============
inline void BloomfilterScalarFun(DataChunk &args, ExpressionState &, Vector &result) {
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		return StringVector::AddString(result, "Bloomfilter " + name.GetString() + " 🐥");
	});
}
inline void BloomfilterOpenSSLVersionScalarFun(DataChunk &args, ExpressionState &, Vector &result) {
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		return StringVector::AddString(result, "Bloomfilter " + name.GetString() + ", my linked OpenSSL version is " +
		                                           OPENSSL_VERSION_TEXT);
	});
}

// =======================
// Extension registration
// =======================
static void LoadInternal(DatabaseInstance &instance) {
	ExtensionUtil::RegisterFunction(instance,
		ScalarFunction("bloom_calc_m",
			{LogicalType::BIGINT, LogicalType::DOUBLE}, LogicalType::INTEGER, BloomCalcMScalar));

	ExtensionUtil::RegisterFunction(instance,
		ScalarFunction("bloom_calc_k",
			{LogicalType::INTEGER, LogicalType::BIGINT, LogicalType::DOUBLE}, LogicalType::INTEGER, BloomCalcKScalar));

	ExtensionUtil::RegisterFunction(instance,
		ScalarFunction("bloom_build",
			{LogicalType::LIST(LogicalType::ANY), LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::BIGINT},
			LogicalType::BLOB, BloomBuildScalar));

	ExtensionUtil::RegisterFunction(instance,
		ScalarFunction("bloom_maybe_contains",
			{LogicalType::BLOB, LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::BIGINT, LogicalType::ANY},
			LogicalType::BOOLEAN, BloomMaybeContains));

	// template demo fns
	ExtensionUtil::RegisterFunction(instance,
		ScalarFunction("bloomfilter", {LogicalType::VARCHAR}, LogicalType::VARCHAR, BloomfilterScalarFun));
	ExtensionUtil::RegisterFunction(instance,
		ScalarFunction("bloomfilter_openssl_version", {LogicalType::VARCHAR}, LogicalType::VARCHAR,
			BloomfilterOpenSSLVersionScalarFun));
}

void BloomfilterExtension::Load(DuckDB &db) { LoadInternal(*db.instance); }
std::string BloomfilterExtension::Name() { return "bloomfilter"; }
std::string BloomfilterExtension::Version() const {
#ifdef EXT_VERSION_BLOOMFILTER
	return EXT_VERSION_BLOOMFILTER;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void bloomfilter_init(duckdb::DatabaseInstance &db) {
	duckdb::DuckDB db_wrapper(db);
	db_wrapper.LoadExtension<duckdb::BloomfilterExtension>();
}

DUCKDB_EXTENSION_API const char *bloomfilter_version() {
	return duckdb::DuckDB::LibraryVersion();
}

}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
