
#define CODETEST 0
#define OPTUNE 0
#define PERFORMANCE 0
#define EVAL 0
#define UNIT_TEST 0


#define TIME_LIMIT (9700)

#define NOT_SUBMIT 0
#define VALIDATION 0

#define IO_FILE 0

#define OUTPUT_INFO 0
#define OUTPUT_FINAL_INFO 0
#define OUTPUT_LOG 0
#define OUTPUT_VISUAL 0

#define FIX_RESULT 0


#define TIME_LIMIT_US (TIME_LIMIT * 1000)

#ifdef __clang_version__
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wmissing-braces"
#endif

#ifndef _MSC_VER 

#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")

#pragma GCC optimize ("O3,inline,omit-frame-pointer,unroll-loops")



#endif

#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
using namespace std;


#define FOR(i, s, e) for (int i = int(s); i < int(e); ++i)
#define RFOR(i, s, e) for (int i = int(e) - 1; i >= int(s); --i)
#define REP(i, n) for (int i = 0, i##_size = int(n); i < i##_size; ++i)
#define RREP(i, n) for (int i = int(n) - 1; i >= 0; --i)


#define ALL(x) (x).begin(),(x).end()

template <class T, class U> inline void chmin(T& a, U&& b) { if (b < a) { a = b; } }
template <class T, class U> inline void chmax(T& a, U&& b) { if (a < b) { a = b; } }
template <class T, class U, class V> inline void clip(T& v, U&& lower, V&& upper) {
	if (v < lower) { v = lower; }
	else if (v > upper) { v = upper; }
}
template <class T> inline constexpr T square(T v) { return v * v; }

template <class T, int SIZE>
constexpr int len(const T(&)[SIZE]) { return SIZE; }

#define cauto const auto

#include <cstdint>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using s8 = int8_t;
using s16 = int16_t;
using s32 = int32_t;
using s64 = int64_t;



struct MemoryException {};


#define VALIDATE_ABORT()
#define VALIDATE_ASSERT(exp)


#define VABORT() VALIDATE_ABORT()
#define VASSERT(exp) VALIDATE_ASSERT(exp)




using TimePoint = chrono::high_resolution_clock::time_point;

struct ChronoTimer {
private:
	TimePoint startTime_;
	TimePoint endTime_;

public:
	inline void Init() {
		startTime_ = chrono::high_resolution_clock::now();
		endTime_ = startTime_;
	}

	inline void Start(int limit) {
		endTime_ = startTime_ + chrono::milliseconds(limit);
	}
	inline void StartMs(int limit) {
		endTime_ = startTime_ + chrono::milliseconds(limit);
	}
	inline void StartUs(int limit) {
		endTime_ = startTime_ + chrono::microseconds(limit);
	}

	inline void Join() {
	}

	inline bool IsOver() const {
		return chrono::high_resolution_clock::now() >= endTime_;
	}

	inline int ElapseTimeMs() const {
		return (int)chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime_).count();
	}
	inline int ElapseTimeUs() const {
		return (int)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime_).count();
	}

	void SetElapseTimeMs(int ms) {
		auto now = chrono::high_resolution_clock::now();
		auto limit = endTime_ - startTime_;
		startTime_ = now - chrono::milliseconds(ms);
		endTime_ = startTime_ + limit;
	}

	inline int LeftToUS(const TimePoint& limit) const {
		return (int)chrono::duration_cast<chrono::microseconds>(limit - chrono::high_resolution_clock::now()).count();
	}

	inline double NowRate() const {
		return (chrono::high_resolution_clock::now() - startTime_).count() / (double)(endTime_ - startTime_).count();
	}

	inline TimePoint Now() const {
		return chrono::high_resolution_clock::now();
	}
	inline TimePoint StartTime() const {
		return startTime_;
	}
	inline TimePoint EndTime() const {
		return endTime_;
	}

	TimePoint GetLimitTimePointUs(int limit) const {
		return startTime_ + chrono::microseconds(limit);
	}
};

TimePoint Now() {
	return chrono::high_resolution_clock::now();
}


template <class T>
void InstanceRun(int argc, const char* argv[]) {
	T* m = new T;
	m->Run(argc, argv);
	quick_exit(0);
}

struct Main;
signed main(int argc, const char* argv[]) {
	cin.tie(0);
	ios::sync_with_stdio(0);
	cout << fixed << setprecision(numeric_limits<double>::digits10);
	cerr << fixed << setprecision(numeric_limits<double>::digits10);
	InstanceRun<Main>(argc, argv);
}




template <class A, class B>
struct pr {
	union {
		A a;
		A x;
		A first;
	};
	union {
		B b;
		B y;
		B second;
	};

	bool operator == (pr const& r) const { return a == r.a && b == r.b; }
	bool operator != (pr const& r) const { return !((*this) == r); }
	bool operator < (pr const& r) const {
		if (a == r.a) {
			return b < r.b;
		}
		return a < r.a;
	}
	bool operator > (pr const& r) const {
		return r < (*this);
	}


	pr& operator += (pr const& v) {
		a += v.a;
		b += v.b;
		return *this;
	}
	pr& operator -= (pr const& v) {
		a -= v.a;
		b -= v.b;
		return *this;
	}

	template <class C, class D>
	auto operator + (pr<C, D> const& v) const {
		return pr<decltype(a + v.a), decltype(b + v.b)>{ a + v.a, b + v.b };
	}

	template <class C, class D>
	auto operator - (pr<C, D> const& v) const {
		return pr<decltype(a - v.a), decltype(b - v.b)>{ a - v.a, b - v.b };
	}

	template <class C, class D>
	explicit operator pr<C, D>() const {
		return { C(a), D(b) };
	}

	template <class T>
	auto operator * (T const& v) const -> pr<decltype(x * v), decltype(y * v)> {
		return { x * v, y * v };
	}
	template <class T>
	auto operator / (T const& v) const -> pr<decltype(x / v), decltype(y / v)> {
		return { x / v, y / v };
	}

	template <class T>
	pr& operator *= (T const& v) {
		x *= v;
		y *= v;
		return *this;
	}
	template <class T>
	pr& operator /= (T const& v) {
		x /= v;
		y /= v;
		return *this;
	}

	pr operator -() const {
		return pr{ -x, -y };
	}

	void flip() { swap(x, y); }

	friend istream& operator>>(istream& is, pr& p) {
		is >> p.a >> p.b;
		return is;
	}
	friend ostream& operator<<(ostream& os, pr const& p) {
		os << p.a << " " << p.b;
		return os;
	}

	template <size_t I>
	auto get() const {
		if constexpr (I == 0) {
			return x;
		}
		else if constexpr (I == 1) {
			return y;
		}
	}
};
using pint = pr<int, int>;
using pdouble = pr<double, double>;

static_assert(is_trivially_copyable<pint>::value, "not trivially_copyable");

template <class A, class B>
struct tuple_size<pr<A, B>> : integral_constant<size_t, 2> {};

template <class A, class B>
struct tuple_element<0, pr<A, B>> { using type = A; };
template <class A, class B>
struct tuple_element<1, pr<A, B>> { using type = B; };

inline pdouble ToDouble(const pint& p) {
	return pdouble{ double(p.x), double(p.y) };
}
inline pint round(const pdouble& d) {
	return pint{ (int)round(d.x), (int)round(d.y) };
}
inline double norm(const pdouble& d) {
	return sqrt((d.x * d.x) + (d.y * d.y));
}
inline double norm(const pint& d) {
	return norm(ToDouble(d));
}
inline int norm2(const pint& d) {
	return square(d.x) + square(d.y);
}
inline pdouble normalized(const pdouble& d) {
	return d / norm(d);
}
inline double dot(const pdouble& a, const pdouble& b) {
	return a.x * b.x + a.y * b.y;
}
inline double cross(const pdouble& a, const pdouble& b) {
	return a.x * b.y - a.y * b.x;
}



template <class T, int CAP>
class CapArr {
private:
	friend class CapArr;

	static_assert(is_trivially_copyable<T>::value);

	T array_[CAP];
	int size_ = 0;

public:


	bool operator == (const CapArr<T, CAP>& r) const {
		if (size_ != r.size_) {
			return false;
		}
		REP(i, size_) {
			if (!(array_[i] == r.array_[i])) {
				return false;
			}
		}
		return true;
	}
	template <class U, int U_CAP>
	bool operator != (const CapArr<U, U_CAP>& r) const {
		return !(*this == r);
	}

	bool MemEqual(const CapArr<T, CAP>& r) const {
		if (size_ != r.size_) {
			return false;
		}
		return memcmp(data(), r.data(), sizeof(T) * size_) == 0;
	}

	constexpr int capacity() const {
		return CAP;
	}

	int size() const {
		return size_;
	}
	bool empty() const {
		return size_ == 0;
	}

	void clear() {
		size_ = 0;
	}

	void resize(int size) {
		size_ = size;
	}

	void assign(int size, const T& e) {
		size_ = size;
		if constexpr (sizeof(T) == 1) {
			if constexpr (is_enum<T>::value) {
				memset(data(), underlying_type<T>::type(e), size);
			}
			else {
				memset(data(), *reinterpret_cast<const unsigned char*>(&e), size);
			}
		}
		else {
			for (int i = 0; i < size; ++i) {
				array_[i] = e;
			}
		}
	}

	void AssignIota(int size) {
		resize(size);
		iota(begin(), end(), 0);
	}
	void Iota(int size) {
		resize(size);
		iota(begin(), end(), 0);
	}

	void MemAssign(int size, int byte) {
		size_ = size;
		memset(data(), byte, sizeof(T) * size);
	}

	void MemCopy(const CapArr<T, CAP>& from) {
		size_ = from.size_;
		memcpy(data(), from.data(), sizeof(T) * from.size_);
	}

	const T* data() const {
		return &array_[0];
	}
	T* data() {
		return &array_[0];
	}

	T& front() {
		return array_[0];
	}
	const T& front() const {
		return array_[0];
	}

	T& back() {
		return array_[size_ - 1];
	}
	const T& back() const {
		return array_[size_ - 1];
	}

	T& operator[](int index) {
		return array_[index];
	}

	const T& operator[](int index) const {
		return array_[index];
	}

	T* begin() {
		return &array_[0];
	}
	T* end() {
		return &array_[size_];
	}
	const T* begin() const {
		return &array_[0];
	}
	const T* end() const {
		return &array_[size_];
	}

	[[nodiscard]] T& push() {
		auto& ref = array_[size_];
		++size_;
		return ref;
	}
	void push(const T& e) {
		array_[size_] = e;
		++size_;
	}

	void pop() {
		--size_;
	}

	int find(const T& value) const {

		REP(i, size_) {
			if (array_[i] == value) {
				return i;
			}
		}
		return -1;
	}
	bool contains(const T& value) const {
		for (const auto& v : *this) {
			if (v == value) {
				return true;
			}
		}
		return false;
	}

	void insert(int index, const T& value) {
		insert(index, &value, 1);
	}

	void insert(int index, const T* mem, int size) {
		if (index == size_) {
			memcpy(data() + index, mem, sizeof(T) * size);
			size_ += size;
		}
		else {
			memmove(data() + index + size, data() + index, sizeof(T) * (size_ - index));
			memcpy(data() + index, mem, sizeof(T) * size);
			size_ += size;
		}
	}

	template <int RCAP>
	void append(const CapArr<T, RCAP>& r) {
		insert(size(), r.data(), r.size());
	}

	void remove(int index) {
		remove(index, index + 1);
	}

	void remove(int start, int end) {
		int size = end - start;
		memmove(data() + start, data() + end, sizeof(T) * (size_ - end));
		size_ -= size;
	}

	void RemoveSwap(int index) {
		array_[index] = array_[size_ - 1];
		--size_;
	}

	void RemoveInsert(int start, int end, const T* p, int size) {
		int newEnd = start + size;
		if (size_ - end > 0 && newEnd != end) {
			memmove(data() + newEnd, data() + end, sizeof(T) * (size_ - end));
		}

		memcpy(data() + start, p, sizeof(T) * size);

		size_ -= end - start;
		size_ += size;
	}

	template <class LESS>
	void stable_sort(LESS&& less) {
		::stable_sort(begin(), end(), less);
	}

};




template <class T, int CAPACITY>
struct CapacityQueue {
private:
	array<T, CAPACITY> ar_ = {};
	int start_ = 0;
	int end_ = 0;

public:
	inline void clear() {
		start_ = 0;
		end_ = 0;
	}

	inline void push(const T& v) {
		ar_[end_] = v;
		++end_;
	}

	inline T* push() {
		T* ptr = &ar_[end_];
		++end_;
		return ptr;
	}

	inline const T& get() const  {
		return ar_[start_];
	}

	inline T pop() {
		return ar_[start_++];
	}

	inline bool empty() const {
		return start_ == end_;
	}

	inline bool exist() const {
		return start_ != end_;
	}

	inline int size() const {
		return end_ - start_;
	}

	inline int total_push_count() const {
		return end_;
	}

	const T& operator[](int i) const{
		return ar_[i];
	}
	int end_size() const {
		return end_;
	}
	int direct_start() const {
		return start_;
	}
	int direct_end() const {
		return end_;
	}

	inline auto begin() -> decltype(ar_.begin()) {
		return ar_.begin() + start_;
	}
	inline auto end() -> decltype(ar_.begin()) {
		return ar_.begin() + end_;
	}
	inline auto begin() const -> decltype(ar_.begin()) {
		return ar_.begin() + start_;
	}
	inline auto end() const -> decltype(ar_.begin()) {
		return ar_.begin() + end_;
	}

	const T& front() const {
		return ar_[start_];
	}
	const T& back() const {
		return ar_[end_ - 1];
	}


};

template <class T, int CAPACITY>
using CapQue = CapacityQueue<T, CAPACITY>;






template <int S>
struct CheckMapS {
private:
    array<u32, S> checked_ = {};
    u32 mark_ = 1;

public:
    void Clear() {
        ++mark_;

        if (mark_ == 0) {
            checked_ = {};
            ++mark_;
        }
    }
    bool IsChecked(int i) const {
        return checked_[i] == mark_;
    }
    void Check(int i) {
        checked_[i] = mark_;
    }
    void Reset(int i) {
        checked_[i] = mark_ - 1;
    }
    bool operator[](int i) const {
        return checked_[i] == mark_;
    }

    bool operator == (const CheckMapS<S>& r) const {
        REP(i, S) {
            if (this->IsChecked(i) != r.IsChecked(i)) {
                return false;
            }
        }
        return true;
    }
};

template <class T, int S>
struct CheckMapDataS {
private:
    array<T, S> data_;
    array<u32, S> checked_ = {};
    u32 mark_ = 1;

public:
    void Clear() {
        ++mark_;

        if (mark_ == 0) {
            checked_ = {};
            ++mark_;
        }
    }

    bool IsChecked(int i) const {
        return checked_[i] == mark_;
    }
    void Check(int i) {
        checked_[i] = mark_;
    }

    void Set(int i, const T& value) {
        checked_[i] = mark_;
        data_[i] = value;
    }

    void Reset(int i) {
        checked_[i] = mark_ - 1;
    }
    const T& Get(int i) const {
        VASSERT(checked_[i] == mark_);
        return data_[i];
    }
    T& Ref(int i) {
        VASSERT(checked_[i] == mark_);
        return data_[i];
    }
    const T& Ref(int i) const {
        VASSERT(checked_[i] == mark_);
        return data_[i];
    }
    T& operator[](int i) {
        VASSERT(checked_[i] == mark_);
        return data_[i];
    }
    const T& operator[](int i) const {
        VASSERT(checked_[i] == mark_);
        return data_[i];
    }

    T GetIf(int i, const T& defaultValue) const {
        if (checked_[i] == mark_) {
            return data_[i];
        }
        return defaultValue;
    }
};

template <class T, int CAP>
struct CapacitySet {
private:
	CapArr<T, CAP> elemens;
	CheckMapDataS<T, CAP> indexTable;

public:
	CapacitySet() {
	}

	bool operator == (const CapacitySet<T, CAP>& r) const {
		if (elemens.size() != r.elemens.size()) {
			return false;
		}
		for (int i : elemens) {
			if (!r.IsContain(i)) {
				return false;
			}
		}
		return true;
	}

	constexpr int capacity() {
		return CAP;
	}

	void Fill() {
		indexTable.Clear();
		elemens.resize(CAP);
		iota(ALL(elemens), 0);
		REP(i, CAP) {
			indexTable.Set(i, i);
		}
	}

	void Clear() {
		elemens.clear();
		indexTable.Clear();
	}

	void Add(T ai) {
		indexTable.Set(ai, elemens.size());
		elemens.push(ai);
	}

	void ForceAdd(T ai) {
		if (indexTable.IsChecked(ai)) {
			return;
		}
		indexTable.Set(ai, elemens.size());
		elemens.push(ai);
	}

	void Remove(int ai) {
		T removeIndex = indexTable[ai];
		T lastIndex = elemens.size() - 1;

		if (removeIndex != lastIndex) {
			elemens[removeIndex] = elemens[lastIndex];
			indexTable.Set(elemens[lastIndex], removeIndex);
		}
		elemens.pop();
		indexTable.Reset(ai);
	}

	void ForceRemove(T ai) {
		if (!indexTable.IsChecked(ai)) {
			return;
		}
		T removeIndex = indexTable[ai];
		T lastIndex = elemens.size() - 1;

		if (removeIndex != lastIndex) {
			elemens[removeIndex] = elemens[lastIndex];
			indexTable.Set(elemens[lastIndex], removeIndex);
		}
		elemens.pop();
		indexTable.Reset(ai);
	}

	bool contains(T i) const {
		return indexTable.IsChecked(i);
	}
	bool IsContain(T i) const {
		return contains(i);
	}

	int size() const {
		return elemens.size();
	}
	bool empty() const {
		return elemens.empty();
	}

	T At(int index) const {
		return elemens[index];
	}

	T operator[](int index) const {
		return elemens[index];
	}

	auto begin() -> decltype(elemens.begin()) {
		return elemens.begin();
	}
	auto end() -> decltype(elemens.begin()) {
		return elemens.end();
	}
	auto begin() const -> decltype(elemens.begin()) {
		return elemens.begin();
	}
	auto end() const -> decltype(elemens.begin()) {
		return elemens.end();
	}
};
template <class T, int CAP>
using CapSet = CapacitySet<T, CAP>;




template <class T, int CAP>
struct CapacityMap {
private:
	CapArr<pr<int, T>, CAP> elemens;
	CheckMapDataS<int, CAP> indexTable;

public:
	CapacityMap() {
		indexTable.Clear();
	}

	void Clear() {
		elemens.clear();
		indexTable.Clear();
	}

	inline void Add(int i, const T& value) {
		indexTable.Set(i, elemens.size());
		elemens.push({ i, value });
	}

	inline void ForceAdd(int i, const T& value) {
		if (indexTable.IsChecked(i)) {
			return;
		}
		indexTable.Set(i, elemens.size());
		elemens.push({ i, value });
	}

	inline void Remove(int i) {
		int removeIndex = indexTable[i];
		int lastIndex = elemens.GetCount() - 1;

		if (removeIndex != lastIndex) {
			elemens[removeIndex] = elemens[lastIndex];
			indexTable[elemens[lastIndex].a] = removeIndex;
		}
		elemens.pop();
		indexTable.Reset(i);
	}

	inline void ForceRemove(int i) {
		if (!indexTable.IsChecked(i)) {
			return;
		}
		int removeIndex = indexTable[i];
		int lastIndex = elemens.size() - 1;

		if (removeIndex != lastIndex) {
			elemens[removeIndex] = elemens[lastIndex];
			indexTable[elemens[lastIndex].a] = removeIndex;
		}
		elemens.pop();
		indexTable.Reset(i);
	}

	inline bool IsContain(int i) const {
		return indexTable.IsChecked(i);
	}

	inline int GetCount() const {
		return elemens.size();
	}




	inline auto begin() -> decltype(elemens.begin()) {
		return elemens.begin();
	}
	inline auto end() -> decltype(elemens.begin()) {
		return elemens.end();
	}
	inline auto begin() const -> decltype(elemens.begin()) {
		return elemens.begin();
	}
	inline auto end() const -> decltype(elemens.begin()) {
		return elemens.end();
	}
};


template <class T, int CAPACITY>
using CapMap = struct CapacityMap<T, CAPACITY>;

#include <cstdint>


struct Xor64 {
	using result_type = uint64_t;
	static constexpr result_type min() { return 0; }
	static constexpr result_type max() { return UINT64_MAX; }

private:
	Xor64(const Xor64& r) = delete;
	Xor64& operator =(const Xor64& r) = delete;
public:

	uint64_t x;
	inline Xor64(uint64_t seed = 0) {
		x = 88172645463325252ULL + seed;
	}

	inline uint64_t operator()() {
		x = x ^ (x << 7);
		return x = x ^ (x >> 9);
	}

	inline uint64_t operator()(uint64_t l, uint64_t r) {
		return ((*this)() % (r - l)) + l;
	}

	template <class T>
	inline T operator()(T r) {
		return (*this)() % r;
	}
	inline double GetDouble() {
		return (*this)() / (double)UINT64_MAX;
	}
	inline bool GetProb(double E) {
		return GetDouble() <= E;
	}
};




enum class Dir : int8_t {
	L = 0,
	U,
	R,
	D,
	N,
	Invalid,
};

constexpr array<Dir, 4> Dir4 = {
	Dir::L,
	Dir::U,
	Dir::R,
	Dir::D,
};

constexpr array<pint, 4> Around4 = { pint{-1, 0}, pint{0, -1}, pint{1, 0}, pint{0, 1} };
constexpr array<pint, 5> Around5 = { pint{-1, 0}, pint{0, -1}, pint{1, 0}, pint{0, 1}, pint{0, 0} };

inline Dir RotateRight(Dir d) {
	constexpr Dir nexts[4] = {
		Dir::U,
		Dir::R,
		Dir::D,
		Dir::L,
	};
	return nexts[(int8_t)d];
}
inline Dir RotateLeft(Dir d) {
	constexpr Dir nexts[4] = {
		Dir::D,
		Dir::L,
		Dir::U,
		Dir::R,
	};
	return nexts[(int8_t)d];
}

inline Dir Back(Dir d) {
	return Dir(s8(d) ^ 2);
}

bool IsHorizontal(Dir dir) {
	return dir == Dir::L || dir == Dir::R;
}
bool IsVertical(Dir dir) {
	return dir == Dir::U || dir == Dir::D;
}

inline Dir CalcDir(const pint& from, const pint& to) {
	if (from.x > to.x) {
		return Dir::L;
	}
	else if (from.y > to.y) {
		return Dir::U;
	}
	else if (from.x < to.x) {
		return Dir::R;
	}
	else if (from.y < to.y) {
		return Dir::D;
	}
	else {
		return Dir::N;
	}
}


inline const string& DirString(Dir dir) {
	static const string strs[6] = {
		"LEFT",
		"UP",
		"RIGHT",
		"DOWN",
		"WAIT",
		"INVALID",
	};
	return strs[(int)dir];
}

inline char DirToChar(Dir dir) {
	static const char chars[6] = {
		'L',
		'U',
		'R',
		'D',
		'N',
		'*',
	};
	return chars[(int)dir];
}

inline Dir CharToDir(char c) {
	if (c == 'L') {
		return Dir::L;
	}
	else if (c == 'U') {
		return Dir::U;
	}
	else if (c == 'R') {
		return Dir::R;
	}
	else if (c == 'D') {
		return Dir::D;
	}
	else if (c == 'N') {
		return Dir::N;
	}
	VABORT();
	return Dir::Invalid;
}

struct GridSystemD {
	int W;
	int H;
	int WH;	

private:
	vector<pint> toPos_;

public:
	void Init(int w, int h) {
		W = w;
		H = h;
		WH = W * H;

		toPos_.resize(WH);
		REP(i, WH) {
			toPos_[i].x = i % W;
			toPos_[i].y = i / W;
		}
	}

	inline int ToId(int x, int y) const {
		VASSERT(x >= 0 && x < W);
		VASSERT(y >= 0 && y < H);
		return x + W * y;
	}
	inline int ToId(const pint& p) const {
		VASSERT(p.x >= 0 && p.x < W);
		VASSERT(p.y >= 0 && p.y < H);
		return p.x + W * p.y;
	}
	inline const pint& ToPos(int id) const {
		return toPos_[id];
	}

	inline int CalcL1Dist(const pint& a, const pint& b) const {
		return abs(a.x - b.x) + abs(a.y - b.y);
	}
	inline int CalcL1Dist(int a, int b) const {
		return CalcL1Dist(ToPos(a), ToPos(b));
	}

	inline int CalcL2Dist2(const pint& a, const pint& b) const {
		return square(a.x - b.x) + square(a.y - b.y);
	}
	inline int CalcL2Dist2(int a, int b) const {
		return CalcL2Dist2(ToPos(a), ToPos(b));
	}

	inline double CalcL2Dist(const pint& a, const pint& b) const {
		return sqrt(CalcL2Dist2(a, b));
	}
	inline double CalcL2Dist(int a, int b) const {
		return CalcL2Dist(ToPos(a), ToPos(b));
	}

	inline bool IsOut(int x, int y) const {
		if (x < 0 || x >= W ||
			y < 0 || y >= H) {
			return true;
		}
		return false;
	}
	inline bool IsOut(const pint& p) const {
		return IsOut(p.x, p.y);
	}
	inline bool IsIn(int x, int y) const {
		return !IsOut(x, y);
	}
	inline bool IsIn(const pint& p) const {
		return !IsOut(p.x, p.y);
	}

	inline bool IsBorder(int x, int y) const {
		if (IsOut(x, y)) {
			return false;
		}
		if (x == 0 || x == W - 1 ||
			y == 0 || y == H - 1) {
			return true;
		}
		return false;
	}
	inline bool IsBorder(const pint& p) const {
		return IsBorder(p.x, p.y);
	}
	inline bool IsBorder(int cell) const {
		return IsBorder(ToPos(cell));
	}

	inline int RotateRight90(int id) const {
		pint p = ToPos(id);
		return ToId(W - 1 - p.y, p.x);
	}

	inline Dir CalcDir1(int from, int to) const {
		if (from - 1 == to) {
			return Dir::L;
		}
		else if (from - W == to) {
			return Dir::U;
		}
		else if (from + 1 == to) {
			return Dir::R;
		}
		else if (from + W == to) {
			return Dir::D;
		}
		else {
			VABORT();
		}
		return Dir::Invalid;
	}
	inline Dir CalcDirM(int from, int to) const {
		return CalcDir(ToPos(from), ToPos(to));
	}
};

template <class T, int CAP>
struct CCA {
private:
    T ar[CAP];      
    int s;

public:
    inline constexpr void push(const T& v) {
        ar[s++] = v;
    }
    inline constexpr const T* begin() const {
        return &ar[0];
    }
    inline constexpr const T* end() const {
        return &ar[s];
    }

    inline constexpr const T& operator ()(int i) const {
        VASSERT(i >= 0 && i < CAP);
        return ar[i];
    }
    inline constexpr const T& operator [](int i) const {
        VASSERT(i >= 0 && i < CAP);
        return ar[i];
    }
    inline constexpr int size() const {
        return s;
    }
};

template <int AROUND_COUNT>
struct AroundMapD {
    using CA = CCA<int, AROUND_COUNT>;
    vector<CA> table_;
    int width_ = 1;

    void Init(int width, int height, const array<pint, AROUND_COUNT>& aroundDirs) {
        width_ = width;
        int count = width * height;
        table_.clear();
        table_.resize(count);

        REP(i, count) {
            pint p = { i % width, i / width };
            for (const pint& a : aroundDirs) {
                pint n = p + a;
                if (n.a >= 0 && n.a < width &&
                    n.b >= 0 && n.b < height) {
                    table_[i].push(n.a + n.b * width);
                }
            }
        }
    }

    inline const CA& operator[](int i) const {
        return table_[i];
    }
    inline const CA& operator[](const pint& p) const {
        return table_[p.x + p.y * width_];
    }
};

template <int AROUND_COUNT>
struct DirMapD {
    using CA = CCA<int, AROUND_COUNT>;
    vector<CA> table_;
    int width_ = 1;

    void Init(int W, int H, const array<pint, AROUND_COUNT>& aroundDirs) {
        width_ = W;
        int count = W * H;
        table_.clear();
        table_.resize(count);

        REP(cellId, count) {
            pint p = { cellId % W, cellId / W };
            for (const pint& a : aroundDirs) {
                int x = p.a + a.a;
                int y = p.b + a.b;
                int n = -1;
                if (x >= 0 && x < W &&
                    y >= 0 && y < H) {
                    n = x + y * W;
                }
                table_[cellId].push(n);
            }
        }
    }

    inline const CA& operator ()(int id) const {
        return table_[id];
    }
    inline const CA& operator [](int id) const {
        return table_[id];
    }
};
GridSystemD gs;
AroundMapD<4> aroundMap;
DirMapD<4> dirMap;


constexpr int minN = 8, maxN = 50;         
constexpr int minS = 1, maxS = 5;          
constexpr int minC = 1, maxC = 30;         
constexpr int minP = 1, maxP = 30;         
constexpr int minT = 30, maxT = 90;        
constexpr int minZ = 1, maxZ = 4;          

constexpr int maxNN = maxN * maxN;

template <class T> using NArr = CapArr<T, maxN>;
template <class T> using NNArr = CapArr<T, maxNN>;
template <class T> using SArr = CapArr<T, maxS>;


template <class T> using NNQue = CapQue<T, maxNN>;

template <class T> using NNSet = CapSet<T, maxNN>;


int N;
int ConnectorCost;		
int PipeCost;			
int SprinklerCost;		
int SpraySize;			

struct InputGrid {
	NNArr<s8> grid_;
	SArr<int> sources_;		
	NNArr<int> plants_;		
	NNArr<bool> reachable_;	

	bool IsSource(int p) const {
		return grid_[p] == 1;
	}
	bool IsPlant(int p) const {
		return grid_[p] == 2;
	}

	bool CanPutSprinkler(int p) const {
		return reachable_[p] && grid_[p] == 0;
	}
};
InputGrid Grid;

int NN;
int ZA;			
double D;		

#define PARAM_CATEGORY(NAME, VALUE, ...) int NAME = VALUE;
#define PARAM_INT(NAME, VALUE, LOWER_VALUE, UPPER_VALUE) int NAME = VALUE;
#define PARAM_DOUBLE(NAME, VALUE, LOWER_VALUE, UPPER_VALUE) double NAME = VALUE;


#define PARAM_LOWER(v)
#define PARAM_UPPER(v)
#define START_TUNING
#define END_TUNING

#define PARAM_GROUP(NAME)
#define PARAM_GROUP_END


template <class T>
struct InputParam {
	T minValue_;
	T maxValue_;
	T value_;

	InputParam(T minValue, T maxValue) {
		minValue_ = minValue;
		maxValue_ = maxValue;
	}

	void SetValue(T value) {
		value_ = value;
	}

	double GetRate(double strong) const {
		double r = (value_ - minValue_) / (double)(maxValue_ - minValue_) * 2.0 - 1.0;		
		return r * strong;
	}
};

static double BlendDoubleParam(double baseValue, double minValue, double maxValue, initializer_list<double> rates) {
	double totalRate = 1;
	for (double rate : rates) {
		totalRate += rate;
	}

	double value = baseValue * totalRate;

	chmax(value, minValue);
	chmin(value, maxValue);

	return value;
}

static int BlendIntParam(double baseValue, int minValue, int maxValue, initializer_list<double> rates) {
	double totalRate = 1;
	for (double rate : rates) {
		totalRate += rate;
	}

	int value = (int)round(baseValue * totalRate);

	chmax(value, minValue);
	chmin(value, maxValue);

	return value;
}



InputParam<int> ipNN(minN * minN, maxNN);
InputParam<int> ipS(minS, maxS);
InputParam<int> ipC(minC, maxC);
InputParam<int> ipP(minP, maxP);
InputParam<int> ipT(minT, maxT);
InputParam<int> ipZA(4, 48);
InputParam<double> ipD(0.05, 0.3);		

constexpr
struct {








	PARAM_DOUBLE(StartTemp, 8.883581341425144, 8.5, 9.5);PARAM_LOWER(0.0);
	PARAM_DOUBLE(EndTemp, 0.22996563643454393, 0.15, 0.25);PARAM_LOWER(0.0);
	PARAM_DOUBLE(InitialStateCount, 5.864066911134709, 5.0, 6.0);PARAM_LOWER(1.0);
	PARAM_DOUBLE(StartTemp_P, 0.24183260576135646, 0.2, 0.25);
	PARAM_DOUBLE(StartTemp_T, 0.10157798207916237, 0.1, 0.15);
	PARAM_DOUBLE(StartTemp_D, 0.10195485187228043, 0.08, 0.12);
	PARAM_DOUBLE(EndTemp_S, 0.21187688997496784, 0.15, 0.25);
	PARAM_DOUBLE(EndTemp_D, -0.2559138283328044, -0.28, -0.22);
	PARAM_DOUBLE(InitialStateCount_P, 0.17750734641274335, 0.14, 0.18);
	PARAM_DOUBLE(InitialStateCount_T, -0.27607115187842185, -0.32, -0.28);
	PARAM_INT(RollbackCount, 11002, 11000, 13000);PARAM_LOWER(0);
	PARAM_DOUBLE(RollbackMulti, 0.9729964293462025, 0.92, 1.0);PARAM_LOWER(0.5);

	START_TUNING;
	PARAM_INT(MinRollbackCount, 351, 1, 500);PARAM_LOWER(1);PARAM_UPPER(500);
	END_TUNING;

	PARAM_DOUBLE(StartTemp_NN, 0.0312383550685841, 0.0, 0.05);
	PARAM_DOUBLE(StartTemp_S, 0.002285863351538399, -0.01, 0.01);
	PARAM_DOUBLE(StartTemp_C, 0.0029937569170513686, -0.01, 0.01);
	PARAM_DOUBLE(StartTemp_ZA, -0.09589498400662011, -0.15, -0.05);
	PARAM_DOUBLE(EndTemp_NN, 0.002030734391409445, -0.01, 0.01);
	PARAM_DOUBLE(EndTemp_C, 0.0058595804229545434, 0.0, 0.05);
	PARAM_DOUBLE(EndTemp_P, 0.0921375675478019, 0.05, 0.15);
	PARAM_DOUBLE(EndTemp_T, 0.07020488804033856, 0.05, 0.1);
	PARAM_DOUBLE(EndTemp_ZA, -0.004550087838051679, -0.01, 0.01);
	PARAM_DOUBLE(InitialStateCount_NN, 0.08657384902902747, 0.05, 0.15);
	PARAM_DOUBLE(InitialStateCount_S, -0.08969259578390568, -0.15, -0.05);
	PARAM_DOUBLE(InitialStateCount_C, -0.000916544914558066, -0.01, 0.01);
	PARAM_DOUBLE(InitialStateCount_ZA, 0.07693474279974803, 0.0, 0.1);
	PARAM_DOUBLE(InitialStateCount_D, -0.0031264679793836644, -0.01, 0.01);




	PARAM_INT(Trans_Remove1Sprinkler1Branch, 16, 2, 6);PARAM_LOWER(0);
	PARAM_INT(Trans_Remove1Sprinkler2Branch, 11, 2, 6);PARAM_LOWER(0);
	PARAM_INT(Trans_Remove1Sprinkler3Branch, 0, 0, 3);PARAM_LOWER(0);
	PARAM_INT(Trans_Remove1Branch, 0, 1, 5);PARAM_LOWER(0);
	PARAM_INT(Trans_Remove2Branch, 0, 0, 3);PARAM_LOWER(0);
	PARAM_INT(Trans_RemoveNear2Sprinkler, 5, 0, 3);PARAM_LOWER(0);
	PARAM_INT(Trans_RemoveNear3Sprinkler, 0, 0, 3);PARAM_LOWER(0);
	PARAM_INT(Trans_RemoveNear4Sprinkler, 0, 0, 3);PARAM_LOWER(0);
	PARAM_GROUP_END;

} HP;



struct IOCommand {
	char type_;		
	int cellA_;		
	int cellB_;		

	void SetPipe(int from, int to) {
		type_ = 'P';
		cellA_ = from;
		cellB_ = to;
	}

	void SetSprinkler(int pos) {
		type_ = 'S';
		cellA_ = pos;
	}

	void Output(ostream& os) const {
		if (type_ == 'P') {
			pint a = gs.ToPos(cellA_);
			pint b = gs.ToPos(cellB_);
			os << type_ << " " << a.y << " " << a.x << " " << b.y << " " << b.x << endl;
		}
		else if (type_ == 'S') {
			pint a = gs.ToPos(cellA_);
			os << type_ << " " << a.y << " " << a.x << endl;
		}
		else {
			VABORT();
		}
	}
};

struct IOResult {
	NNArr<IOCommand> commands_;

	IOCommand& push() {
		return commands_.push();
	}
	void Output(ostream& os) const {
		os << commands_.size() << endl;
		REP(i, commands_.size()) {
			commands_[i].Output(os);
		}
	}
};

struct SprayMap {
    using CA = CCA<int, 49>;
    vector<CA> table_;
    int width_ = 1;

    void Init(int width, int height, int spraySize) {
        width_ = width;
        int count = width * height;
        table_.clear();
        table_.resize(count);

        int sp2 = spraySize * spraySize;

        CCA<pint, 49> offsets = {};
        FOR(oy, -4, 5) {
            FOR(ox, -4, 5) {
                if (oy * oy + ox * ox <= sp2) {
                    offsets.push(pint{ ox, oy });
                }
            }
        }

        REP(i, count) {
            pint p = gs.ToPos(i);
            for (cauto& off : offsets) {
                pint n = p + off;
                if (gs.IsOut(n)) {
                    continue;
                }
                table_[i].push(gs.ToId(n));
            }
        }
    }

    inline const CA& operator[](int i) const {
        return table_[i];
    }
    inline const CA& operator[](const pint& p) const {
        return table_[p.x + p.y * width_];
    }
};

struct SprinklerPlantMap {
    using CA = CCA<int, 48>;
    vector<CA> table_;
    int width_ = 1;

    void Init(int width, int height, int spraySize) {
        width_ = width;
        int count = width * height;
        table_.clear();
        table_.resize(count);

        int sp2 = spraySize * spraySize;

        CCA<pint, 48> offsets = {};
        FOR(oy, -spraySize, spraySize + 1) {
            FOR(ox, -spraySize, spraySize + 1) {
                if (ox == 0 && oy == 0) {
                    continue;
                }
                if (oy * oy + ox * ox <= sp2) {
                    offsets.push(pint{ ox, oy });
                }
            }
        }

        REP(i, count) {
            if (!Grid.CanPutSprinkler(i)) {
                continue;
            }
            pint p = gs.ToPos(i);
            for (cauto& off : offsets) {
                pint n = p + off;
                if (gs.IsOut(n)) {
                    continue;
                }
                if (Grid.IsPlant(gs.ToId(n))) {
                    table_[i].push(gs.ToId(n));
                }
            }
        }
    }

    inline const CA& operator[](int i) const {
        return table_[i];
    }
    inline const CA& operator[](const pint& p) const {
        return table_[p.x + p.y * width_];
    }
};

struct PlantSprinklerMap {
    using CA = CCA<int, 48>;
    vector<CA> table_;
    int width_ = 1;

    void Init(int width, int height, int spraySize) {
        width_ = width;
        int count = width * height;
        table_.clear();
        table_.resize(count);

        int sp2 = spraySize * spraySize;

        CCA<pint, 48> offsets = {};
        FOR(oy, -spraySize, spraySize + 1) {
            FOR(ox, -spraySize, spraySize + 1) {
                if (ox == 0 && oy == 0) {
                    continue;
                }
                if (oy * oy + ox * ox <= sp2) {
                    offsets.push(pint{ ox, oy });
                }
            }
        }

        REP(i, count) {
            if (!Grid.IsPlant(i)) {
                continue;
            }
            pint p = gs.ToPos(i);
            for (cauto& off : offsets) {
                pint n = p + off;
                if (gs.IsOut(n)) {
                    continue;
                }
                if (Grid.CanPutSprinkler(gs.ToId(n))) {
                    table_[i].push(gs.ToId(n));
                }
            }
        }
    }

    inline const CA& operator[](int i) const {
        return table_[i];
    }
    inline const CA& operator[](const pint& p) const {
        return table_[p.x + p.y * width_];
    }
};


struct AroundEmptyMap {
    using CA = CCA<int, 4>;
    NNArr<CA> table_;
    int width_ = 1;

    void Init(int width, int height) {
        width_ = width;
        int count = width * height;
        table_.clear();
        table_.resize(count);

        REP(i, count) {
            pint p = { i % width, i / width };
            for (const pint& a : Around4) {
                pint n = p + a;
                if (n.a >= 0 && n.a < width &&
                    n.b >= 0 && n.b < height) {
                    if (!Grid.IsPlant(gs.ToId(n))) {
                        table_[i].push(n.a + n.b * width);
                    }
                }
            }
        }
    }

    inline const CA& operator[](int i) const {
        return table_[i];
    }
    inline const CA& operator[](const pint& p) const {
        return table_[p.x + p.y * width_];
    }
};

struct AroundEmptyDirMap {
    using CA = CCA<pr<int, Dir>, 4>;
    NNArr<CA> table_;
    int width_ = 1;

    void Init(int width, int height) {
        width_ = width;
        int count = width * height;
        table_.clear();
        table_.resize(count);

        REP(i, count) {
            pint p = { i % width, i / width };
            REP(di, 4) {
                const pint& a = Around4[di];
                pint n = p + a;
                if (n.a >= 0 && n.a < width &&
                    n.b >= 0 && n.b < height) {
                    if (!Grid.IsPlant(gs.ToId(n))) {
                        table_[i].push(pr<int, Dir>{n.a + n.b * width, Dir(di)});
                    }
                }
            }
        }
    }

    inline const CA& operator[](int i) const {
        return table_[i];
    }
    inline const CA& operator[](const pint& p) const {
        return table_[p.x + p.y * width_];
    }
};

struct RandAroundEmptyMap {
    using CA = CCA<int, 4>;
    NNArr<CapArr<CA, 24>> table_;
    int width_ = 1;

    void Init(int width, int height) {
        width_ = width;
        int count = width * height;
        table_.clear();
        table_.resize(count);

        REP(i, count) {
            pint p = { i % width, i / width };
            CA ca = {};
            for (const pint& a : Around4) {
                pint n = p + a;
                if (n.a >= 0 && n.a < width &&
                    n.b >= 0 && n.b < height) {
                    if (!Grid.IsPlant(gs.ToId(n))) {
                        ca.push(n.a + n.b * width);
                    }
                }
            }

            CapArr<int, 4> idxs;
            idxs.Iota(ca.size());
            do {
                auto& t = table_[i].push();
                for (int idx : idxs) {
                    t.push(ca[idx]);
                }
            } while (next_permutation(ALL(idxs)));
        }
    }

    const CA& operator()(int i, Xor64& rand_) const {
        return table_[i][rand_(table_[i].size())];
    }

};
SprinklerPlantMap sprinklerPlantMap;
PlantSprinklerMap plantSprinklerMap;
AroundEmptyMap aroundEmptyMap;      
AroundEmptyDirMap aroundEmptyDirMap;      



constexpr double LinearParam(double ax, double ay, double bx, double by, double cx) {
	double r = (cx - ax) / (bx - ax);
	double cy = ay + (by - ay) * r;
	return cy;
}

constexpr double LinearParam2(
	double left, double right, double top, double bottom,
	double valueLT, double valueRT, double valueLB, double valueRB,
	double x, double y) {
	double rx = (x - left) / (right - left);
	double ry = (y - top) / (bottom - top);

	double dst = (1 - rx) * (1 - ry) * valueLT
		+ (1 - rx) * ry * valueLB
		+ rx * (1 - ry) * valueRT
		+ rx * ry * valueRB;
	return dst;
}

int LinearParamInt(double ax, double ay, double bx, double by, double cx) {
	return (int)round(LinearParam(ax, ay, bx, by, cx));
}
int LinearParam2Int(
	double left, double right, double top, double bottom,
	double valueLT, double valueRT, double valueLB, double valueRB,
	double x, double y) {
	return (int)round(LinearParam2(left, right, top, bottom,
		valueLT, valueRT, valueLB, valueRB,
		x, y));
}



struct PipeGrid {
	struct Bit {
		u8 pipe : 7;
		u8 sprinkler : 1;

		bool operator == (const Bit& r) const {
			return pipe == r.pipe &&
				sprinkler == r.sprinkler;
		}

	};
	NNArr<Bit> bits_;		

	bool operator == (const PipeGrid& r) const {
		return bits_ == r.bits_;
	}

	void init() {
		bits_.assign(NN, Bit{});
	}

	void SetPipe(int p, Dir dir) {
		int n = dirMap[p][(int)dir];
		bits_[p].pipe |= (1 << (int)dir);
		bits_[n].pipe |= (1 << (int)Back(dir));
	}
	void SetPipe(int p, int n) {
		SetPipe(p, gs.CalcDir1(p, n));
	}
	void RemovePipe(int p, Dir dir) {
		int n = dirMap[p][(int)dir];
		bits_[p].pipe &= ~(1 << (int)dir);
		bits_[n].pipe &= ~(1 << (int)Back(dir));
	}
	void RemovePipe(int p, int n) {
		RemovePipe(p, gs.CalcDir1(p, n));
	}
	bool IsPipe(int p, Dir dir) const {
		return (bits_[p].pipe >> (int)dir) & 1;
	}
	bool IsPipe(int p, int n) const {
		return IsPipe(p, gs.CalcDir1(p, n));
	}

	bool IsExistPipe(int p) const {
		return bits_[p].pipe != 0;
	}

	void SetSprinkler(int p) {
		bits_[p].sprinkler = true;
	}
	void RemoveSprinkler(int p) {
		bits_[p].sprinkler = false;
	}
	bool IsSprinkler(int p) const {
		return bits_[p].sprinkler;
	}

	int GetConnect(int p) const {
		if (Grid.IsSource(p)) {
			return 0;
		}
		u8 b = bits_[p].pipe;
		if (b == 5 || b == 10) {
			return 0;
		}
		return ((b & 1) ? 1 : 0) + ((b & 2) ? 1 : 0) + ((b & 4) ? 1 : 0) + ((b & 8) ? 1 : 0);
	}

	int GetEdgeFrom(int p) const {
		u8 b = bits_[p].pipe;
		VASSERT(GetConnect(p) == 1);
		if (b == 1) {
			return dirMap[p][(int)Dir::L];
		}
		else if (b == 2) {
			return dirMap[p][(int)Dir::U];
		}
		else if (b == 4) {
			return dirMap[p][(int)Dir::R];
		}
		else if (b == 8) {
			return dirMap[p][(int)Dir::D];
		}
		else {
			VABORT();
		}
		return -1;
	}

	int CalcScore(NNArr<bool>& water, NNArr<bool>& spray, array<int, 5>& connectors, int& connect, int& pipe, int& sprinkler, int& dryPlant) const {
		{
			water.assign(NN, false);
			NNQue<int> que;
			for (int source : Grid.sources_) {
				que.push(source);
				water[source] = true;
			}
			while (que.exist()) {
				int p = que.pop();
				for (int n : aroundMap[p]) {
					if (IsPipe(p, n)) {
						if (!water[n]) {
							water[n] = true;
							que.push(n);
						}
					}
				}
			}
		}

		spray.assign(NN, false);
		REP(i, NN) {
			if (IsSprinkler(i) && water[i]) {
				for (int n : sprinklerPlantMap[i]) {
					spray[n] = true;
				}
			}
		}

		connectors = {};
		connect = 0;
		pipe = 0;
		sprinkler = 0;
		dryPlant = 0;
		REP(i, NN) {
			if (Grid.IsPlant(i)) {
				if (!spray[i]) {
					++dryPlant;
				}
			}
			else if (IsSprinkler(i)) {
				++sprinkler;
			}

			if (bits_[i].pipe) {
				if (bits_[i].pipe & 1) {
					++pipe;
				}
				if (bits_[i].pipe & 2) {
					++pipe;
				}

				int c = GetConnect(i);
				if (c > 0) {
					++connectors[c];
					++connect;
				}
			}
		}

		int score = connectors[1] * ConnectorCost
			+ connectors[2] * ConnectorCost * 2
			+ connectors[3] * ConnectorCost * 3
			+ connectors[4] * ConnectorCost * 4
			+ pipe * PipeCost
			+ sprinkler * SprinklerCost
			+ dryPlant * NN;
		return score;
	}

	int CalcScore() const {
		NNArr<bool> water;
		NNArr<bool> spray;
		array<int, 5> connectors = {};
		int connect = 0;
		int pipe = 0;
		int sprinkler = 0;
		int dryPlant = 0;
		return CalcScore(water, spray, connectors, connect, pipe, sprinkler, dryPlant);
	}

	void MakeResult(IOResult& result) {
		REP(i, NN) {
			for (Dir dir : {Dir::R, Dir::D}) {
				if (IsPipe(i, dir)) {
					result.push().SetPipe(i, dirMap[i][(int)dir]);
				}
			}
		}
		REP(i, NN) {
			if (IsSprinkler(i)) {
				result.push().SetSprinkler(i);
			}
		}
	}
};





struct IOServer {
	double scoreScale_ = 1;		

	void InitInput(ChronoTimer& timer) {
		istream& is = cin;
		is >> N;
		timer.Init();		

		NN = N * N;

		is >> ConnectorCost >> PipeCost >> SprinklerCost >> SpraySize;
		Grid.grid_.resize(NN);
		REP(i, NN) {
			int v;
			is >> v;
			Grid.grid_[i] = v;

			if (v == 1) {
				Grid.sources_.push(i);
			}
			if (v == 2) {
				Grid.plants_.push(i);
			}
		}

		gs.Init(N, N);
		aroundMap.Init(N, N, Around4);
		dirMap.Init(N, N, Around4);

		{
			NNArr<bool>& reachable = Grid.reachable_;
			reachable.assign(NN, false);

			NNQue<int> que;
			for (int source : Grid.sources_) {
				reachable[source] = true;
				que.push(source);
			}
			while (que.exist()) {
				int p = que.pop();
				for (int n : aroundMap[p]) {
					if (Grid.IsPlant(n)) {
						continue;
					}
					if (reachable[n]) {
						continue;
					}
					reachable[n] = true;
					que.push(n);
				}
			}
		}


		sprinklerPlantMap.Init(N, N, SpraySize);		
		aroundEmptyMap.Init(N, N);
		aroundEmptyDirMap.Init(N, N);

		{
			NNArr<bool>& reachable = Grid.reachable_;
			REP(i, NN) {
				if (reachable[i]) {
					int minusCost = sprinklerPlantMap[i].size() * NN;		
					if (minusCost <= SprinklerCost) {
						reachable[i] = false;
					}
				}
			}
		}

		sprinklerPlantMap.Init(N, N, SpraySize);		
		plantSprinklerMap.Init(N, N, SpraySize);

		constexpr array<int, 5> ZATbl = {0, 4, 12, 28, 48};
		ZA = ZATbl[SpraySize];
		D = Grid.plants_.size() / (double)NN;

		constexpr double NN_coef[2] = { -7.19113923341608, 1.01523904633906 };
		constexpr double S_coef[2] = { -0.115685172452688, -0.0452922753462569 };
		constexpr double C_coef[2] = { 132.381365658751, 2.24076041450801 };
		constexpr double P_coef[2] = { 14.0690111302367, 0.781555440046337 };
		constexpr double T_coef[2] = { 0.12496243170102, 0.369509314754529 };
		constexpr double ZA_coef[2] = { 2.40430134745659E-05, 0.522689008501489 };
		constexpr double D_coef[2] = { 0.21006081037281, 1.26747404947356 };
		constexpr double scale_coef[2] = { 48977666.7682826, 0 };
		int S = Grid.sources_.size();
		int C = ConnectorCost;
		int P = PipeCost;
		int T = SprinklerCost;
		scoreScale_ = scale_coef[0];
		scoreScale_ /= pow(NN + NN_coef[0], NN_coef[1]);
		scoreScale_ /= pow(S + S_coef[0], S_coef[1]);
		scoreScale_ /= pow(C + C_coef[0], C_coef[1]);
		scoreScale_ /= pow(P + P_coef[0], P_coef[1]);
		scoreScale_ /= pow(T + T_coef[0], T_coef[1]);
		scoreScale_ *= pow(ZA + ZA_coef[0], ZA_coef[1]);
		scoreScale_ /= pow(D + D_coef[0], D_coef[1]);

		ipNN.SetValue(NN);
		ipS.SetValue(S);
		ipC.SetValue(C);
		ipP.SetValue(P);
		ipT.SetValue(T);
		ipZA.SetValue(ZA);
		ipD.SetValue(D);

	}

	void Output(const IOResult& result) {
		ostream& os = cout;
		result.Output(os);
	}

	void Finalize() {
	}
};
IOServer server;

template <class T, int CAP>
struct CapPriorityQueue {
    CapArr<T, CAP> buf_;

    void clear() {
        buf_.clear();
    }

    bool empty() const {
        return buf_.empty();
    }
    bool exist() const {
        return !buf_.empty();
    }

    const T& top() {
        return buf_.front();
    }

    template <class CMP>
    void push(const T& v, CMP&& cmp) {
        buf_.push(v);
        push_heap(ALL(buf_), cmp);
    }

    template <class CMP>
    T pop(CMP&& cmp) {
        pop_heap(ALL(buf_), cmp);
        T ret = buf_.back();
        buf_.pop();
        return ret;
    }

    void push(const T& v) {
        buf_.push(v);
        push_heap(ALL(buf_));
    }

    void pop() {
        pop_heap(ALL(buf_));
        buf_.pop();
    }



    void PushNoHeap(const T& v) {
        buf_.push(v);
    }
    void MakeHeap() {
        make_heap(ALL(buf_));
    }

};


template <class IT, class RAND>
void Shuffle(IT&& begin, IT&& end, RAND&& rand) {
	int size = int(end - begin);
	if (size <= 1) {
		return;
	}
	REP(i, size - 1) {
		int j = i + rand() % (size - i);
		swap(*(begin + i), *(begin + j));
	}
}


struct RandomTable {
	vector<int> table_;

	void push(int value, int count) {
		table_.reserve(table_.size() + count);
		REP(i, count) {
			table_.emplace_back(value);
		}
	}

	template <class ENGINE>
	int operator()(ENGINE& engine) {
		return table_[engine() % table_.size()];
	}
};

#define VISUAL_SA 0

#define USE_SA_POINT_FILTER 1
#define USE_SA_ROLLBACK 1

#define USE_ACCEPT_SCORE 1



struct SAChecker {
	Xor64* rand_ = nullptr;
	double* totalMaxScore_ = nullptr;

	double temp = 0;		
	double divTemp = 0;

	double currentScore = 0;
	double maxScore = 0;

	int noMaxUpdateCount = 0;				
	int nextRollbackCheckCount = INT_MAX;	

	inline bool operator()(double newScore, bool forceUpdate = false) {
		++noMaxUpdateCount;

		if (newScore > currentScore) {
			currentScore = newScore;
			if (newScore > maxScore) {
				maxScore = newScore;
				noMaxUpdateCount = 0;

				if (newScore > *totalMaxScore_) {
					*totalMaxScore_ = newScore;
				}
			}

			return true;
		}

		else if (newScore == currentScore) {
			return true;
		}

		else {
			if (forceUpdate || exp((newScore - currentScore) * divTemp) * UINT32_MAX > (*rand_)(UINT32_MAX)) {
				currentScore = newScore;
				return true;
			}
			else {
				return false;
			}
		}
	}

	double AcceptScore() {
		static_assert(numeric_limits<double>::is_iec559);
		return currentScore + temp * log(rand_->GetDouble());
	}

	void SetRevert() {
		++noMaxUpdateCount;

	}

};

template <class F>
struct SATransition {
	const char* name;
	F func;
	int weight;
};
template <class F>
auto MakeTransition(const char* name, F&& func, int weight) {
	return SATransition<F>{ name, func, weight };
}
#define MAKE_TRANS(func, weight) MakeTransition(#func, [&](SAChecker& sac, State& state) { func(sa, sac, state); }, weight)

struct SimulatedAnnealing {
	vector<SAChecker> checkers;

	double totalMaxScore = 0;				
	double timeRate = 0;				
	double temp = 0;					
	double divTemp = 0;					


	Xor64 rand_;

	double startTemp_ = 200;					
	double endTemp_ = 1;						
	int stepLoopCount = 1000;					

	double rollbackStartRate_ = 999.0;			
	int rollbackCount_ = INT_MAX;				
	double rollbackNextMulti_ = 1.1;			
	int minRollbackCount_ = 1;					


public:
	template <class STATE, class... TRANSITION>
	void Run2(ChronoTimer& timer, vector<STATE>& states, tuple<SATransition<TRANSITION>...>& transitions) {

		vector<STATE> maxStates = states;
		checkers.resize(states.size());
		totalMaxScore = states[0].EvalScore();
		REP(pi, checkers.size()) {
			auto& checker = checkers[pi];
			checker.rand_ = &rand_;
			checker.totalMaxScore_ = &totalMaxScore;

			checker.temp = 0;
			checker.divTemp = 0;
			checker.currentScore = states[pi].EvalScore();
			checker.maxScore = checker.currentScore;
			checker.noMaxUpdateCount = 0;
			checker.nextRollbackCheckCount = rollbackCount_;

			chmax(totalMaxScore, checker.maxScore);
		}

		RandomTable randTable;
		TupleLoop(transitions, [&](auto&& e, size_t i) {
			randTable.push((int)i, e.weight);
		});

		const auto startTime = timer.Now();
		const auto endTime = timer.EndTime();
		const double subTimeCountDiv = 1.0 / (double)(endTime - startTime).count();

		vector<int> pis(states.size());
		iota(ALL(pis), 0);

		bool forceEnd = false;
		while (!timer.IsOver()) {
			timeRate = (timer.Now() - startTime).count() * subTimeCountDiv;
			temp = startTemp_ * std::pow(endTemp_ / startTemp_, timeRate);		
			divTemp = 1.0 / temp;
			for (auto& checker : checkers) {
				checker.temp = temp;
				checker.divTemp = divTemp;
			}


			for (int rp = 0; rp < stepLoopCount; ++rp) {
				int ti = (int)randTable(rand_);

				auto exec = [&](auto&& e, size_t i) {
					for (int pi : pis) {
						auto& checker = checkers[pi];
						e.func(checker, states[pi]);

						if (states[pi].RawScore() > maxStates[pi].RawScore()) {
							maxStates[pi] = states[pi];
						}
						else {
							if (timeRate >= rollbackStartRate_) {
								if (checker.noMaxUpdateCount >= checker.nextRollbackCheckCount) {
									states[pi] = maxStates[pi];
									checker.noMaxUpdateCount = 0;
									checker.nextRollbackCheckCount = (int)round(checker.nextRollbackCheckCount * rollbackNextMulti_);
									chmax(checker.nextRollbackCheckCount, minRollbackCount_);
								}
							}
						}
					}
				};

				TupleAccess(transitions, ti, exec);
			}
			if (forceEnd) {
				break;
			}

			{
				constexpr double start = 0.2;
				constexpr double end = 1.0;
				int targetPointCount = (int)states.size();
				if (timeRate >= end) {
					targetPointCount = 1;
				}
				else if (timeRate >= start) {
					double r = 1.0 - (timeRate - start) / (end - start);		
					targetPointCount = 1 + (int)floor(states.size() * r);
				}
				if ((int)pis.size() > targetPointCount) {
					sort(ALL(pis), [&](int a, int b) {
						return checkers[a].maxScore > checkers[b].maxScore;
					});
					pis.resize(targetPointCount);
				}
			}
		}

	}

	void ForceUpdate() {
	}

private:
	template <class Tuple, class Func>
	void TupleLoop(Tuple & t, Func && f) {
		TupleLoop2(t, forward<Func>(f), make_index_sequence<tuple_size<Tuple>::value>{});
	}
	template <class Tuple, class Func, size_t... Indics>
	void TupleLoop2(Tuple & t, Func && f, index_sequence<Indics...>) {
		using swallow = int[];
		(void)swallow {
			(TupleLoop3<Tuple, Func, Indics>(t, f), 0)...
		};
	}
	template <class Tuple, class Func, size_t Index>
	void TupleLoop3(Tuple & t, Func & f) {
		f(get<Index>(t), Index);
	}

	template <class Tuple, class Func>
	void TupleAccess(Tuple & t, int i, Func && f) {
		TupleAccess2(t, i, forward<Func>(f), make_index_sequence<tuple_size<Tuple>::value>{});
	}
	template <class Tuple, class Func, size_t... Indics>
	void TupleAccess2(Tuple & t, int i, Func && f, index_sequence<Indics...>) {
		using swallow = int[];
		(void)swallow {
			(TupleAccess3<Tuple, Func, Indics>(t, i, f), 0)...
		};
	}
	template <class Tuple, class Func, size_t Index>
	void TupleAccess3(Tuple & t, int i, Func & f) {
		if (i == Index) {
			f(get<Index>(t), Index);
		}
	}

};


void MakeGreedy(PipeGrid& grid) {
	NNQue<int> que;
	NNArr<int> dists;
	NNArr<int> froms;
	dists.assign(NN, -1);
	froms.assign(NN, -1);
	for (int source : Grid.sources_) {
		dists[source] = 0;
		que.push(source);
	}
	while (que.exist()) {
		int p = que.pop();
		for (int n : aroundMap[p]) {
			if (dists[n] >= 0) {
				continue;
			}

			dists[n] = dists[p] + 1;
			froms[n] = p;

			if (Grid.IsPlant(n)) {
				continue;
			}

			que.push(n);
		}
	}

	NNArr<int> ps;
	ps.Iota(NN);
	ps.stable_sort([&](int a, int b) {
		return dists[a] > dists[b];
		});

	NNArr<bool> isCheckedPlant;
	isCheckedPlant.assign(NN, false);

	grid.init();

	for (int p : ps) {
		if (Grid.IsPlant(p)) {
			if (isCheckedPlant[p]) {
				continue;
			}

			int sp = -1;
			for (int n : plantSprinklerMap[p]) {
				if (dists[n] >= 0) {
					if (sp < 0 || dists[n] < dists[sp]) {
						if (Grid.CanPutSprinkler(n)) {
							sp = n;
						}
					}
				}
			}
			if (sp < 0) {
				continue;
			}

			grid.SetSprinkler(sp);
			for (int n : sprinklerPlantMap[sp]) {
				isCheckedPlant[n] = true;
			}

			while (true) {
				int n = froms[sp];
				if (n < 0) {
					break;
				}
				if (!Grid.IsPlant(sp)) {
					grid.SetPipe(sp, n);
				}
				sp = n;
			}
		}
	}
}

Xor64 rand_;

int totalProcessCount = 0;

struct StateBackup {
	CapMap<bool, maxNN * 2> pipeDir_;	
	CapMap<bool, maxNN> sprinkler_;		
	int rawScore_;						

	void Init() {
		pipeDir_.Clear();
		sprinkler_.Clear();
		rawScore_ = -1;
	}

	void BackupRawScore(int rawScore) {
		if (rawScore_ < 0) {
			rawScore_ = rawScore;
		}
	}

	int PipeDir2Index(int p, int n) const {
		Dir dir = gs.CalcDir1(p, n);
		if (dir == Dir::R) {
			++p;
			dir = Dir::L;
		}
		else if (dir == Dir::D) {
			p += N;
			dir = Dir::U;
		}
		return p * 2 + (int)dir;
	}
	void Index2PipeDir(int index, int& p, Dir& dir) const {
		p = index / 2;
		dir = Dir(index & 1);
	}

	void BackupPipe(int p, int n, bool exist) {
		int index = PipeDir2Index(p, n);
		if (!pipeDir_.IsContain(index)) {
			pipeDir_.Add(index, exist);
		}
	}

	void BackupSprinkler(int p, bool exist) {
		if (!sprinkler_.IsContain(p)) {
			sprinkler_.Add(p, exist);
		}
	}

};


struct Index {
	union {
		s32 idx;
		struct {
			u32 di : 1;
			u32 p: 31;
		};
	};

	operator int() const {
		return idx;
	}
};
constexpr int IndexSize = sizeof(Index);

Index ToIndex(int p, int di) {
	Index idx;
	idx.p = p;
	idx.di = di;
	return idx;
}
Index MakeInvalidIndex() {
	Index idx;
	idx.idx = -1;
	return idx;
}

struct Elem {
	int cost;		
	Index idx;

	int P() const {
		VASSERT(idx >= 0);
		return idx.p;
	}
	int Di() const {
		VASSERT(idx >= 0);
		return idx.di;
	}

	bool operator < (Elem const& n) const {
		return cost > n.cost;
	}
};

struct Router {
	CheckMapDataS<int, maxNN*2> costs_;
	CheckMapDataS<Index, maxNN*2> froms_;
	CapPriorityQueue<Elem, maxNN * 4> que_;
	CapArr<Elem, maxNN * 2> record_;
	int recordIndex_ = -1;

	void Init(int start) {

		VASSERT(!Grid.IsPlant(start));

		auto& costs = costs_;
		auto& froms = froms_;
		costs.Clear();
		froms.Clear();

		auto& que = que_;
		que.clear();

		{
			Elem elem = Elem{ 0, ToIndex(start, 0) };	
			que.push(elem);
			costs.Set(elem.idx, elem.cost);
			froms.Set(elem.idx, MakeInvalidIndex());
		}
		recordIndex_ = 0;
	}

	bool IsInited() const {
		return recordIndex_ >= 0;
	}

	void Ready() {
		recordIndex_ = 0;
	}

	bool GetNext(Elem& e) {
		if (recordIndex_ < record_.size()) {
			e = record_[recordIndex_];
			++recordIndex_;
			return true;
		}

		auto& costs = costs_;
		auto& froms = froms_;
		auto& que = que_;

		while (!que.empty()) {
			Elem elem = que.top();
			que.pop();
			int p = elem.P();

			if (costs[elem.idx] != elem.cost) {
				continue;
			}

			record_.push(elem);

			for (int n : aroundEmptyMap[p]) {
				if (froms[elem.idx] >= 0 && n == froms[elem.idx].p) {
					continue;
				}

				Elem next;
				next.idx = ToIndex(n, IsHorizontal(gs.CalcDir1(p, n)) ? 0 : 1);
				next.cost = elem.cost + PipeCost;		

				if (froms[elem.idx] < 0) {
					if (Grid.IsSource(p)) {
					}
					else {
						next.cost += ConnectorCost;
					}
				}
				else {
					if (elem.Di() != next.Di()) {
						next.cost += ConnectorCost * 2;
					}
				}

				if (!costs.IsChecked(next.idx) || next.cost < costs[next.idx]) {
					costs.Set(next.idx, next.cost);
					froms.Set(next.idx, elem.idx);
					que.push(next);
				}
			}

			break;
		}

		if (recordIndex_ < record_.size()) {
			e = record_[recordIndex_];
			++recordIndex_;
			return true;
		}
		return false;
	}

};

struct State {
	PipeGrid grid_;
	NNSet<int> sprinklers_;				
	NNSet<int> putCandidates_;			
	NNArr<s8> plantSprayedCounts_;			
	NNArr<s8> requireSprinklerCounts_;		

	int rawScore_;		

	bool operator == (const State& r) const {
		return
			rawScore_ == r.rawScore_ &&
			grid_ == r.grid_ &&
			sprinklers_ == r.sprinklers_ &&
			putCandidates_ == r.putCandidates_ &&
			plantSprayedCounts_ == r.plantSprayedCounts_ &&
			requireSprinklerCounts_ == r.requireSprinklerCounts_;
	}

	void Init() {
		grid_.init();

		sprinklers_.Clear();
		putCandidates_.Clear();
		plantSprayedCounts_.assign(NN, 0);
		requireSprinklerCounts_.assign(NN, 0);
		REP(i, NN) {
			if (Grid.IsPlant(i)) {
				for (int n : plantSprinklerMap[i]) {
					++requireSprinklerCounts_[n];
					if (requireSprinklerCounts_[n] == 1) {
						putCandidates_.Add(n);
					}
				}
			}
		}

		rawScore_ = grid_.CalcScore();
	}

	int RawScore() const {
		return -rawScore_;
	}
	double EvalScore() const {
		return -rawScore_ * server.scoreScale_;
	}

	void RemoveSprinkler(int p, StateBackup& backup) {
		backup.BackupRawScore(rawScore_);
		backup.BackupSprinkler(p, true);

		sprinklers_.Remove(p);
		grid_.RemoveSprinkler(p);
		rawScore_ -= SprinklerCost;

		for (int n : sprinklerPlantMap[p]) {
			--plantSprayedCounts_[n];
			VASSERT(plantSprayedCounts_[n] >= 0);
			if (plantSprayedCounts_[n] == 0) {
				rawScore_ += NN;

				for (int m : plantSprinklerMap[n]) {
					++requireSprinklerCounts_[m];
					if (requireSprinklerCounts_[m] == 1) {
						putCandidates_.Add(m);
					}
				}
			}
		}
	}

	void RemovePipe(int p, int n, StateBackup& backup) {
		VASSERT(grid_.IsPipe(p, n));

		backup.BackupRawScore(rawScore_);

		int prevConnectP = grid_.GetConnect(p);
		int prevConnectN = grid_.GetConnect(n);

		backup.BackupPipe(p, n, true);
		grid_.RemovePipe(p, n);
		rawScore_ -= PipeCost;

		int nextConnectP = grid_.GetConnect(p);
		int nextConnectN = grid_.GetConnect(n);

		rawScore_ += (nextConnectP + nextConnectN - prevConnectP - prevConnectN) * ConnectorCost;

	}

	void RemovePipe(int p, StateBackup& backup) {
		backup.BackupRawScore(rawScore_);

		int centerPrevConnect = grid_.GetConnect(p);
		for (int n : aroundEmptyMap[p]) {
			if (grid_.IsPipe(p, n)) {
				backup.BackupPipe(p, n, true);

				rawScore_ -= PipeCost;

				if (!Grid.IsSource(n)) {
					int prevConnect = grid_.GetConnect(n);		
					grid_.RemovePipe(p, n);
					int nextConnect = grid_.GetConnect(n);		
					rawScore_ += (nextConnect - prevConnect) * ConnectorCost;
				}
			}
		}
		int centerNextConnect = grid_.GetConnect(p);
		rawScore_ += (centerNextConnect - centerPrevConnect) * ConnectorCost;

	}

	void RemovePipeWithBranch(int p, int branchCount, StateBackup& backup) {
		VASSERT(branchCount > 0);
		backup.BackupRawScore(rawScore_);

		NNArr<int> counts;
		counts.assign(NN, -1);

		NNQue<int> que;
		que.push(p);
		counts[p] = 0;

		while (que.exist()) {
			int p = que.pop();
			for (int n : aroundEmptyMap[p]) {
				if (grid_.IsPipe(p, n)) {
					int connect = grid_.GetConnect(n);

					RemovePipe(p, n, backup);

					counts[n] = counts[p];
					if (connect >= 3) {
						++counts[n];
					}

					if (!grid_.IsSprinkler(n) && !Grid.IsSource(n) && counts[n] < branchCount) {
						que.push(n);
					}
				}
			}
		}

	}

	void RemovePipeWithSprinkler(int p, int sprinklerCount, StateBackup& backup) {
		VASSERT(sprinklerCount > 0);
		backup.BackupRawScore(rawScore_);

		NNArr<int> counts;
		counts.assign(NN, -1);

		NNQue<int> que;
		que.push(p);
		counts[p] = 0;

		while (que.exist()) {
			int p = que.pop();
			for (int n : aroundEmptyMap[p]) {
				if (grid_.IsPipe(p, n)) {
					int connect = grid_.GetConnect(n);

					RemovePipe(p, n, backup);

					counts[n] = counts[p];
					if (grid_.IsSprinkler(n)) {
						++counts[n];
					}

					if (!Grid.IsSource(n) && counts[n] < sprinklerCount) {
						que.push(n);
					}
				}
			}
		}

	}

	void RemovePipeWithDistance(int p, int dist, StateBackup& backup) {
		VASSERT(dist > 0);
		backup.BackupRawScore(rawScore_);

		NNArr<int> dists;
		dists.assign(NN, -1);

		NNQue<int> que;
		que.push(p);
		dists[p] = 0;

		while (que.exist()) {
			int p = que.pop();
			for (int n : aroundEmptyMap[p]) {
				if (grid_.IsPipe(p, n)) {
					int connect = grid_.GetConnect(n);

					RemovePipe(p, n, backup);

					dists[n] = dists[p] + 1;

					if (!Grid.IsSource(n) && dists[n] < dist) {
						que.push(n);
					}
				}
			}
		}

	}

	void RemovePipeWithPipe(int p, int pipe, StateBackup& backup) {
		VASSERT(pipe > 0);
		backup.BackupRawScore(rawScore_);

		int removedPipe = 0;

		NNQue<int> que;
		que.push(p);

		while (que.exist()) {
			int p = que.pop();
			for (int n : aroundEmptyMap[p]) {
				if (grid_.IsPipe(p, n)) {
					int connect = grid_.GetConnect(n);

					RemovePipe(p, n, backup);

					++removedPipe;
					if (removedPipe >= pipe) {
						break;
					}

					if (!Grid.IsSource(n)) {
						que.push(n);
					}
				}
			}
			if (removedPipe >= pipe) {
				break;
			}
		}

	}

	void RemoveBranch(int start, int branchCount, StateBackup& backup) {
		VASSERT(branchCount > 0);
		backup.BackupRawScore(rawScore_);

		NNArr<int> counts;
		counts.assign(NN, -1);

		NNQue<int> que;
		que.push(start);
		counts[start] = 0;

		while (que.exist()) {
			int p = que.pop();
			for (int n : aroundEmptyMap[p]) {
				if (grid_.IsPipe(p, n)) {
					int connect = grid_.GetConnect(n);

					RemovePipe(p, n, backup);

					counts[n] = counts[p];
					if (connect >= 3) {
						++counts[n];
					}

					if (!Grid.IsSource(n) && counts[n] < branchCount) {
						que.push(n);
					}
				}
			}

			if (grid_.IsSprinkler(p)) {
				RemoveSprinkler(p, backup);
			}
		}

	}

	void AddSprinkler(int p, StateBackup& backup) {
		backup.BackupRawScore(rawScore_);
		backup.BackupSprinkler(p, false);

		sprinklers_.Add(p);
		grid_.SetSprinkler(p);
		rawScore_ += SprinklerCost;

		for (int n : sprinklerPlantMap[p]) {
			++plantSprayedCounts_[n];
			VASSERT(plantSprayedCounts_[n] > 0);
			if (plantSprayedCounts_[n] == 1) {
				rawScore_ -= NN;

				for (int m : plantSprinklerMap[n]) {
					--requireSprinklerCounts_[m];
					VASSERT(requireSprinklerCounts_[m] >= 0);
					if (requireSprinklerCounts_[m] == 0) {
						putCandidates_.Remove(m);
					}
				}
			}
		}
	}

	void FillSprinkler(StateBackup& backup) {
		while (putCandidates_.size()) {
			int p = putCandidates_[rand_(putCandidates_.size())];
			AddSprinkler(p, backup);
		}
	}

	void RemoveUnnecessarySprinkler(StateBackup& backup) {
		auto IsRequire = [&](int p) {
			for (int n : sprinklerPlantMap[p]) {
				if (plantSprayedCounts_[n] == 1) {
					return true;
				}
			}
			return false;
		};

		NNArr<int> notRequires;
		for (int p : sprinklers_) {
			if (!IsRequire(p)) {
				notRequires.push(p);
			}
		}
		Shuffle(ALL(notRequires), rand_);

		while (true) {
			bool updated = false;
			for (int p : notRequires) {
				if (grid_.IsSprinkler(p)) {
					if (!IsRequire(p)) {
						RemoveSprinkler(p, backup);
						RemovePipeWithBranch(p, 1, backup);
						updated = true;
					}
				}
			}
			if (!updated) {
				break;
			}
		}
	}

	bool Connect(StateBackup& backup, double acceptEvalScore) {
		backup.BackupRawScore(rawScore_);

		if (EvalScore() <= acceptEvalScore) {
			return false;
		}

		static NNArr<NNArr<int>> groupsCells;		
		static NNArr<int> groupMap;		
		groupsCells.clear();
		groupMap.clear();

		{
			groupMap.assign(NN, -1);

			{
				NNQue<int> que;

				{
					int gi = groupsCells.size();
					auto& cells = groupsCells.push();
					cells.clear();

					que.clear();
					for (int source : Grid.sources_) {
						que.push(source);
						cells.push(source);
						groupMap[source] = gi;
					}
					while (que.exist()) {
						int p = que.pop();
						for (auto&&[n, dir] : aroundEmptyDirMap[p]) {
							if (!grid_.IsPipe(p, dir) || groupMap[n] >= 0) {
								continue;
							}
							que.push(n);
							cells.push(n);
							groupMap[n] = gi;
						}
					}
				}

				for (int sp : sprinklers_) {
					if (groupMap[sp] >= 0) {
						continue;
					}

					int gi = groupsCells.size();
					auto& cells = groupsCells.push();
					cells.clear();

					que.clear();
					{
						que.push(sp);
						cells.push(sp);
						groupMap[sp] = gi;
					}
					while (que.exist()) {
						int p = que.pop();
						for (auto&& [n, dir] : aroundEmptyDirMap[p]) {
							if (!grid_.IsPipe(p, dir) || groupMap[n] >= 0) {
								continue;
							}
							que.push(n);
							cells.push(n);
							groupMap[n] = gi;
						}
					}
				}
			}

			if (groupsCells.size() <= 1) {
				return true;
			}
		}

		REP(mergeLoop, groupsCells.size() - 1) {

			static CheckMapDataS<int, maxNN * 2> costs;
			static CheckMapDataS<Index, maxNN * 2> froms;		
			costs.Clear();
			froms.Clear();

			static CapPriorityQueue<Elem, maxNN * 4> que;
			que.clear();

			int bestCellCount = INT_MAX;
			NNArr<int> startGis;
			REP(gi, groupsCells.size()) {
				if (groupsCells[gi].empty()) {
					continue;
				}
				if (groupsCells[gi].size() < bestCellCount) {
					bestCellCount = groupsCells[gi].size();
					startGis.clear();
					startGis.push(gi);
				}
				else if (groupsCells[gi].size() == bestCellCount) {
					startGis.push(gi);
				}
			}
			VASSERT(startGis.size() > 0);
			int startGi = startGis[rand_(startGis.size())];
			VASSERT(startGi >= 0);

			Elem best = { INT_MAX, -1 };
			if (groupsCells[startGi].size() == 1) {
				static NNArr<Router> routers;
				if (routers.empty()) {
					routers.resize(NN);
				}

				int startP = groupsCells[startGi][0];
				auto& r = routers[startP];
				if (!r.IsInited()) {
					r.Init(startP);
				}
				r.Ready();
				{
					Index startIdx = ToIndex(startP, 0);

					Elem e;
					bool ok = r.GetNext(e);
					VASSERT(ok);
					VASSERT(startP == e.P());
					VASSERT(e.Di() == 0);
					costs.Set(startIdx, 0);
					froms.Set(startIdx, MakeInvalidIndex());
				}

				Elem next;
				while (r.GetNext(next)) {
					if (next.cost - ConnectorCost >= best.cost) {
						break;
					}
					Index fromIdx = r.froms_[next.idx];
					if (costs.IsChecked(fromIdx) && costs[fromIdx] == 0) {
					}
					else {
						continue;
					}

					int p = fromIdx.p;
					int n = next.P();

					bool goal = false;
					if (n != startP && (Grid.IsSource(n) || grid_.IsSprinkler(n) || grid_.IsExistPipe(n))) {
						if (Grid.IsSource(n)) {
						}
						else {
							int prevConnect = grid_.GetConnect(n);
							VASSERT(!grid_.IsPipe(p, n));
							grid_.SetPipe(p, n);
							int newConnect = grid_.GetConnect(n);
							grid_.RemovePipe(p, n);
							next.cost += (newConnect - prevConnect) * ConnectorCost;		
						}
						goal = true;
					}

					froms.Set(next.idx, fromIdx);

					if (goal) {
						if (next.cost < best.cost) {
							best = next;
						}
					}
					else {
						costs.Set(next.idx, 0);
					}
				}
			}
			else {
				for (int p : groupsCells[startGi]) {
					Elem elem = Elem{ 0, ToIndex(p, 0) };	
					que.PushNoHeap(elem);
					costs.Set(elem.idx, elem.cost);
					froms.Set(elem.idx, MakeInvalidIndex());
				}

				while (!que.empty()) {
					const Elem elem = que.top();
					que.pop();

					if (costs[elem.idx] != elem.cost) {
						continue;
					}

					if (elem.cost + PipeCost - ConnectorCost * 2 >= best.cost) {
						break;
					}
					int p = elem.P();
					int di = elem.Di();
					for (auto&& [n, dir] : aroundEmptyDirMap[p]) {
						if ((froms[elem.idx] >= 0 && n == froms[elem.idx].p) || groupMap[n] == startGi) {
							continue;
						}

						Elem next;
						next.idx = ToIndex(n, IsHorizontal(dir) ? 0 : 1);
						next.cost = elem.cost + PipeCost;		

						if (froms[elem.idx] < 0) {
							if (Grid.IsSource(p)) {
							}
							else {
								int prevConnect = grid_.GetConnect(p);
								grid_.SetPipe(p, n);
								int newConnect = grid_.GetConnect(p);
								grid_.RemovePipe(p, n);
								next.cost += (newConnect - prevConnect) * ConnectorCost;		
							}
						}
						else {
							if (elem.Di() != next.Di()) {
								next.cost += ConnectorCost * 2;
							}
						}

						bool goal = false;
						if (Grid.IsSource(n) || grid_.IsSprinkler(n) || grid_.IsExistPipe(n)) {
							if (Grid.IsSource(n)) {
							}
							else {
								int prevConnect = grid_.GetConnect(n);
								VASSERT(!grid_.IsPipe(p, n));
								grid_.SetPipe(p, n);
								int newConnect = grid_.GetConnect(n);
								grid_.RemovePipe(p, n);
								next.cost += (newConnect - prevConnect) * ConnectorCost;		
							}
							goal = true;
						}

						if (!costs.IsChecked(next.idx) || next.cost < costs[next.idx]) {
							costs.Set(next.idx, next.cost);
							froms.Set(next.idx, elem.idx);

							if (goal) {
								if (next.cost < best.cost) {
									best = next;
								}
							}
							else {
									que.push(next);
							}
						}
					}
				}
			}

			VASSERT(best.P() >= 0);

			if (-(rawScore_ + best.cost) * server.scoreScale_ <= acceptEvalScore) {
				return false;
			}

			Index idx = best.idx;
			while (true) {
				Index ndx = froms[idx];
				if (ndx < 0) {
					break;
				}
				int c = idx.p;
				int n = ndx.p;
				if (groupMap[c] < 0) {
					groupsCells[startGi].push(c);
					groupMap[c] = startGi;
				}
				backup.BackupPipe(c, n, false);
				grid_.SetPipe(c, n);
				idx = ndx;
			}

			if (mergeLoop + 1 < groupsCells.size()) {
				int connectGi = groupMap[best.P()];
				VASSERT(connectGi >= 0);

				for (int p : groupsCells[connectGi]) {
					VASSERT(groupMap[p] == connectGi);
					groupsCells[startGi].push(p);
					groupMap[p] = startGi;
				}

				groupsCells[connectGi].clear();

			}

			rawScore_ += best.cost;
		}


		return true;
	}

	void Revert(const StateBackup& backup) {
		for (cauto& kv : backup.pipeDir_) {
			int index = kv.a;
			bool exist = kv.b;
			int p = -1;
			Dir dir = Dir::Invalid;
			backup.Index2PipeDir(index, p, dir);
			if (exist) {
				grid_.SetPipe(p, dir);
			}
			else {
				grid_.RemovePipe(p, dir);
			}
		}

		for (cauto& kv : backup.sprinkler_) {
			int p = kv.a;
			bool exist = kv.b;
			if (exist) {
				if (!sprinklers_.IsContain(p)) {
					sprinklers_.Add(p);
					grid_.SetSprinkler(p);

					for (int n : sprinklerPlantMap[p]) {
						++plantSprayedCounts_[n];
						VASSERT(plantSprayedCounts_[n] > 0);
						if (plantSprayedCounts_[n] == 1) {
							for (int m : plantSprinklerMap[n]) {
								--requireSprinklerCounts_[m];
								VASSERT(requireSprinklerCounts_[m] >= 0);
								if (requireSprinklerCounts_[m] == 0) {
									putCandidates_.Remove(m);
								}
							}
						}
					}
				}
			}
			else {
				if (sprinklers_.IsContain(p)) {
					sprinklers_.Remove(p);
					grid_.RemoveSprinkler(p);

					for (int n : sprinklerPlantMap[p]) {
						--plantSprayedCounts_[n];
						VASSERT(plantSprayedCounts_[n] >= 0);
						if (plantSprayedCounts_[n] == 0) {
							for (int m : plantSprinklerMap[n]) {
								++requireSprinklerCounts_[m];
								if (requireSprinklerCounts_[m] == 1) {
									putCandidates_.Add(m);
								}
							}
						}
					}

				}
			}
		}

		if (backup.rawScore_ >= 0) {
			rawScore_ = backup.rawScore_;
		}

	}
};
static constexpr int StateSize = sizeof(State);

struct Solver {


	PipeGrid bestGrid_;
	int bestRawScore_ = INT_MAX;

	void CheckMaxScore(const State& state) {
		if (state.rawScore_ < bestRawScore_) {
			bestRawScore_ = state.rawScore_;
			bestGrid_ = state.grid_;
		}
	}

	void Run(ChronoTimer& timer) {
		{
		}

		vector<State> states;
		InitState(states);

		static double StartTemp = BlendDoubleParam(HP.StartTemp, 0.0, 1e10, {
			ipNN.GetRate(HP.StartTemp_NN),
			ipS.GetRate(HP.StartTemp_S),
			ipC.GetRate(HP.StartTemp_C),
			ipP.GetRate(HP.StartTemp_P),
			ipT.GetRate(HP.StartTemp_T),
			ipZA.GetRate(HP.StartTemp_ZA),
			ipD.GetRate(HP.StartTemp_D),
			});
		static double EndTemp = BlendDoubleParam(HP.EndTemp, 0.0, 1e10, {
			ipNN.GetRate(HP.EndTemp_NN),
			ipS.GetRate(HP.EndTemp_S),
			ipC.GetRate(HP.EndTemp_C),
			ipP.GetRate(HP.EndTemp_P),
			ipT.GetRate(HP.EndTemp_T),
			ipZA.GetRate(HP.EndTemp_ZA),
			ipD.GetRate(HP.EndTemp_D),
			});

		SimulatedAnnealing sa;
		sa.startTemp_ = StartTemp;
		sa.endTemp_ = EndTemp;
		sa.stepLoopCount = 100;

		sa.rollbackStartRate_ = 0.1;
		sa.rollbackCount_ = HP.RollbackCount;		
		sa.rollbackNextMulti_ = HP.RollbackMulti;
		sa.minRollbackCount_ = HP.MinRollbackCount;

		if (N <= 12) {
			auto transitions = make_tuple(
				MAKE_TRANS(Transition_MinN, 1)
			);
			sa.Run2(timer, states, transitions);
		}
		else {
			auto transitions = make_tuple(
				MAKE_TRANS(Transition, 1)
			);
			sa.Run2(timer, states, transitions);
		}
		IOResult result;
		bestGrid_.MakeResult(result);
		server.Output(result);

	}

	void InitState(vector<State>& states) {
		static int InitialStateCount = BlendIntParam(HP.InitialStateCount, 1, 256, {
			ipNN.GetRate(HP.InitialStateCount_NN),
			ipS.GetRate(HP.InitialStateCount_S),
			ipC.GetRate(HP.InitialStateCount_C),
			ipP.GetRate(HP.InitialStateCount_P),
			ipT.GetRate(HP.InitialStateCount_T),
			ipZA.GetRate(HP.InitialStateCount_ZA),
			ipD.GetRate(HP.InitialStateCount_D),
			});

		states.resize(InitialStateCount);
		{
			State& state = states[0];
			state.Init();
		}
		FOR(i, 1, InitialStateCount) {
			states[i] = states[0];
		}

	}

	void Transition(SimulatedAnnealing& sa, SAChecker& checker, State& state) {

		static StateBackup backup;
		backup.Init();

		if (state.putCandidates_.size()) {
			state.FillSprinkler(backup);
		}
		else {
			static RandomTable patternTbl;
			if (patternTbl.table_.empty()) {

				patternTbl.push(0, HP.Trans_Remove1Sprinkler1Branch);
				patternTbl.push(1, HP.Trans_Remove1Sprinkler2Branch);
				patternTbl.push(2, HP.Trans_Remove1Sprinkler3Branch);
				patternTbl.push(3, HP.Trans_Remove1Branch);
				patternTbl.push(4, HP.Trans_Remove1Branch);
				patternTbl.push(5, HP.Trans_RemoveNear2Sprinkler);
				patternTbl.push(6, HP.Trans_RemoveNear3Sprinkler);
				patternTbl.push(7, HP.Trans_RemoveNear4Sprinkler);
			}

			int ptn = patternTbl(rand_);
			if (ptn == 0) {
				int removeP = state.sprinklers_[rand_(state.sprinklers_.size())];
				state.RemoveSprinkler(removeP, backup);
				state.RemovePipeWithBranch(removeP, 1, backup);
			}
			else if (ptn == 1) {
				int removeP = state.sprinklers_[rand_(state.sprinklers_.size())];
				state.RemoveSprinkler(removeP, backup);
				state.RemovePipeWithBranch(removeP, 2, backup);
			}
			else if (ptn == 2) {
				int removeP = state.sprinklers_[rand_(state.sprinklers_.size())];
				state.RemoveSprinkler(removeP, backup);
				state.RemovePipeWithBranch(removeP, 3, backup);
			}
			else if (ptn == 3) {
				int removeP = state.sprinklers_[rand_(state.sprinklers_.size())];
				state.RemoveBranch(removeP, 1, backup);
			}
			else if (ptn == 4) {
				int removeP = state.sprinklers_[rand_(state.sprinklers_.size())];
				state.RemoveBranch(removeP, 2, backup);
			}
			else {
				VASSERT(ptn == 5 || ptn == 6 || ptn == 7);
				int removeP = state.sprinklers_[rand_(state.sprinklers_.size())];
				state.RemoveSprinkler(removeP, backup);
				state.RemovePipeWithBranch(removeP, 1, backup);

				if (state.sprinklers_.size()) {
					NNArr<int> ps;
					int areaSize = square(SpraySize * 3);
					for (int p : state.sprinklers_) {
						int d = gs.CalcL2Dist2(removeP, p);
						if (d <= areaSize) {
							ps.push(p);
						}
					}
					ps.stable_sort([&](int a, int b) {
						return gs.CalcL2Dist2(removeP, a) < gs.CalcL2Dist2(removeP, b);
						});

					int loop = 1;
					if (ptn == 5) {
						loop = 1;
					}
					else if (ptn == 6) {
						loop = 2;
					}
					else {
						VASSERT(ptn == 7);
						loop = 3;
					}

					REP(ri, loop) {
						if (ps.empty()) {
							break;
						}

						int idx = 0;
						int p = ps[idx];
						state.RemoveSprinkler(p, backup);
						state.RemovePipeWithBranch(p, 1, backup);

						ps.remove(idx);
					}
				}
			}

			state.FillSprinkler(backup);
		}

		state.RemoveUnnecessarySprinkler(backup);

		VASSERT(state.putCandidates_.size() == 0);

		double acceptEvalScore = checker.AcceptScore();

		++totalProcessCount;
		bool ok = state.Connect(backup, acceptEvalScore);
		if (ok) {
			VASSERT(state.EvalScore() >= acceptEvalScore);
			checker(state.EvalScore(), true);
			CheckMaxScore(state);

		}
		else {
			checker.SetRevert();
			state.Revert(backup);
		}
	}

	void Transition_MinN(SimulatedAnnealing& sa, SAChecker& checker, State& state) {

		static StateBackup backup;
		backup.Init();

		int pattern = rand_(3);		

		if (pattern == 0 || pattern == 2) {
			if (state.sprinklers_.size()) {
				int removeP = state.sprinklers_[rand_(state.sprinklers_.size())];
				state.RemoveSprinkler(removeP, backup);
				state.RemovePipeWithBranch(removeP, 1 + rand_(2), backup);		
			}
		}

		if (pattern == 1 || pattern == 2) {
			if (state.putCandidates_.size()) {
				int putP = state.putCandidates_[rand_(state.putCandidates_.size())];
				state.AddSprinkler(putP, backup);
			}
		}

		state.RemoveUnnecessarySprinkler(backup);

		double acceptEvalScore = checker.AcceptScore();

		++totalProcessCount;
		bool ok = state.Connect(backup, acceptEvalScore);
		if (ok) {
			VASSERT(state.EvalScore() >= acceptEvalScore);
			checker(state.EvalScore(), true);
			CheckMaxScore(state);
		}
		else {
			checker.SetRevert();
			state.Revert(backup);
		}
	}

};



struct Main {
    void Run(int argc, const char* argv[]) {
        ChronoTimer timer;
        server.InitInput(timer);

        static Solver solver;
        timer.StartMs(TIME_LIMIT);

        solver.Run(timer);

        server.Finalize();

    }
};
