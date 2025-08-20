#define DUCKDB_EXTENSION_MAIN

#include "bloomfilter_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

// OpenSSL linked through vcpkg (left from template; harmless)
#include <openssl/opensslv.h>

namespace duckdb {

// ================================
// Helpers (no external dependencies)
// ================================

static inline bool TestBit(const_data_ptr_t bytes, uint64_t idx) {
	const uint64_t byte = idx >> 3;          // idx / 8
	const uint8_t  mask = 1u << (idx & 7u);  // idx % 8
	return (bytes[byte] & mask) != 0;
}

// Simple 64-bit FNV-1a; we salt the offset basis with 'seed' for variability
static inline uint64_t FNV1a64(const void *data, size_t len, uint64_t seed) {
	const uint8_t *p = static_cast<const uint8_t *>(data);
	uint64_t h = 14695981039346656037ULL ^ seed;         // offset basis ^ seed
	const uint64_t prime = 1099511628211ULL;
	for (size_t i = 0; i < len; i++) {
		h ^= p[i];
		h *= prime;
	}
	return h;
}

static inline void HashFromValue(const Value &v, uint64_t seed, uint64_t &h1, uint64_t &h2) {
	switch (v.type().id()) {
	case LogicalTypeId::TINYINT: {
		int8_t x = v.GetValue<int8_t>();
		h1 = FNV1a64(&x, sizeof(x), seed);
		h2 = FNV1a64(&x, sizeof(x), seed ^ 0x9E3779B97F4A7C15ULL);
		return;
	}
	case LogicalTypeId::SMALLINT: {
		int16_t x = v.GetValue<int16_t>();
		h1 = FNV1a64(&x, sizeof(x), seed);
		h2 = FNV1a64(&x, sizeof(x), seed ^ 0x9E3779B97F4A7C15ULL);
		return;
	}
	case LogicalTypeId::INTEGER: {
		int32_t x = v.GetValue<int32_t>();
		h1 = FNV1a64(&x, sizeof(x), seed);
		h2 = FNV1a64(&x, sizeof(x), seed ^ 0x9E3779B97F4A7C15ULL);
		return;
	}
	case LogicalTypeId::BIGINT: {
		int64_t x = v.GetValue<int64_t>();
		h1 = FNV1a64(&x, sizeof(x), seed);
		h2 = FNV1a64(&x, sizeof(x), seed ^ 0x9E3779B97F4A7C15ULL);
		return;
	}
	case LogicalTypeId::HUGEINT: {
		hugeint_t x = v.GetValue<hugeint_t>();
		h1 = FNV1a64(&x, sizeof(x), seed);
		h2 = FNV1a64(&x, sizeof(x), seed ^ 0x9E3779B97F4A7C15ULL);
		return;
	}
	case LogicalTypeId::FLOAT: {
		float x = v.GetValue<float>();
		if (x == 0.0f) x = 0.0f; // normalize -0.0
		h1 = FNV1a64(&x, sizeof(x), seed);
		h2 = FNV1a64(&x, sizeof(x), seed ^ 0x9E3779B97F4A7C15ULL);
		return;
	}
	case LogicalTypeId::DOUBLE: {
		double x = v.GetValue<double>();
		if (x == 0.0) x = 0.0;   // normalize -0.0
		h1 = FNV1a64(&x, sizeof(x), seed);
		h2 = FNV1a64(&x, sizeof(x), seed ^ 0x9E3779B97F4A7C15ULL);
		return;
	}
	case LogicalTypeId::BLOB:
	case LogicalTypeId::VARCHAR: {
		auto s = StringValue::Get(v);
		h1 = FNV1a64(s.data(), s.size(), seed);
		h2 = FNV1a64(s.data(), s.size(), seed ^ 0x9E3779B97F4A7C15ULL);
		return;
	}
	default: {
		// Fallback for DECIMAL/DATE/TIMESTAMP/etc.
		auto s = v.ToString();
		h1 = FNV1a64(s.data(), s.size(), seed);
		h2 = FNV1a64(s.data(), s.size(), seed ^ 0x9E3779B97F4A7C15ULL);
		return;
	}
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

		const bool bitset_valid = bitset_uf.validity.RowIsValid(bi);
		const bool value_valid  = val_uf.validity.RowIsValid(vi);
		if (!bitset_valid || !value_valid) {
			res[i] = false;
			continue;
		}

		const int32_t m    = ((const int32_t*)m_uf.data)[mi];
		const int32_t k    = ((const int32_t*)k_uf.data)[ki];
		const int64_t seed = ((const int64_t*)seed_uf.data)[si];

		// bitset bytes
		const string_t blob = ((const string_t*)bitset_uf.data)[bi];
		const auto *bytes   = (const_data_ptr_t)blob.GetDataUnsafe();

		// hash value
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

// ==============
// Template demos
// ==============
inline void BloomfilterScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		return StringVector::AddString(result, "Bloomfilter " + name.GetString() + " 🐥");
	});
}

inline void BloomfilterOpenSSLVersionScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
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
	// Register our Bloom probe
	auto bloom_maybe = ScalarFunction(
		"bloom_maybe_contains",
		{LogicalType::BLOB, LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::BIGINT, LogicalType::ANY},
		LogicalType::BOOLEAN,
		BloomMaybeContains
	);
	ExtensionUtil::RegisterFunction(instance, bloom_maybe);

	// Keep template demo functions
	auto bloomfilter_scalar_function =
	    ScalarFunction("bloomfilter", {LogicalType::VARCHAR}, LogicalType::VARCHAR, BloomfilterScalarFun);
	ExtensionUtil::RegisterFunction(instance, bloomfilter_scalar_function);

	auto bloomfilter_openssl_version_scalar_function =
	    ScalarFunction("bloomfilter_openssl_version", {LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                  BloomfilterOpenSSLVersionScalarFun);
	ExtensionUtil::RegisterFunction(instance, bloomfilter_openssl_version_scalar_function);
}

void BloomfilterExtension::Load(DuckDB &db) {
	LoadInternal(*db.instance);
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

extern "C" {

DUCKDB_EXTENSION_API void bloomfilter_init(duckdb::DatabaseInstance &db) {
	duckdb::DuckDB db_wrapper(db);
	db_wrapper.LoadExtension<duckdb::BloomfilterExtension>();
}

DUCKDB_EXTENSION_API const char *bloomfilter_version() {
	// Use the C++ API here (template already does this)
	return duckdb::DuckDB::LibraryVersion();
}

}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
