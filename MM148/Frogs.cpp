
#define CODETEST 0
#define OPTUNE 0
#define PERFORMANCE 0
#define EVAL 0
#define UNIT_TEST 0


#define TIME_LIMIT (9500)

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
				memset(data(), e, size);
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
};

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



#define PARAM_CATEGORY(NAME, VALUE, ...) int NAME = VALUE;
#define PARAM_INT(NAME, VALUE, LOWER_VALUE, UPPER_VALUE) int NAME = VALUE;
#define PARAM_DOUBLE(NAME, VALUE, LOWER_VALUE, UPPER_VALUE) double NAME = VALUE;


#define PARAM_LOWER(v)
#define PARAM_UPPER(v)
#define START_TUNING
#define END_TUNING

#define PARAM_GROUP(NAME)
#define PARAM_GROUP_END




constexpr
struct {









	START_TUNING;
	PARAM_DOUBLE(SameDir2Turn, 1.2371398445014097, 1.0, 1.5);PARAM_LOWER(0.0);
	PARAM_DOUBLE(SameDir3Turn, 1.420619392206073, 1.0, 1.5);PARAM_LOWER(0.0);
	PARAM_DOUBLE(WaterRateHomeTurn, 1.1585033847114978, 0.8, 1.2);PARAM_LOWER(0.0);
	PARAM_DOUBLE(HomeTurnRate, 1.0231640449259314, 0.8, 1.0);PARAM_LOWER(0.0);
	PARAM_DOUBLE(OutTurnRate, 0.631249091204721, 0.4, 0.6);PARAM_LOWER(0.0);
	PARAM_DOUBLE(DepthGeta, 1.0196352040028072, 0.9, 1.1);PARAM_LOWER(0.0);
	PARAM_DOUBLE(DepthRate, 0.38352367763193956, 0.38, 0.43);PARAM_LOWER(0.0);PARAM_UPPER(1.0);
	PARAM_DOUBLE(NoDeadCoinRate, 0.6960848646893455, 0.6, 0.7);PARAM_LOWER(0.0);
	PARAM_DOUBLE(DeadTurnRate, 0.5620753036594475, 0.55, 0.62);PARAM_LOWER(0.0);
	PARAM_DOUBLE(ProbGeta, 0.2535808247615139, 0.25, 0.3);PARAM_LOWER(0.0);PARAM_UPPER(1.0);
	PARAM_DOUBLE(OutFailRate, 0.8710850926583136, 0.85, 0.95);PARAM_LOWER(0.0);
	PARAM_DOUBLE(XScoreRate, 1.3463036555549788, 1.3, 1.4);PARAM_LOWER(0.0);
	PARAM_DOUBLE(FrontScoreRate, 1.5361572956820433, 1.7, 2.3);PARAM_LOWER(0.0);
	PARAM_DOUBLE(YScoreRate, 1.0623790898788903, 1.0, 1.4);PARAM_LOWER(0.0);
	PARAM_DOUBLE(DiffDirRate, 2.8059271911777706, 2.3, 2.7);PARAM_LOWER(0.0);
	PARAM_DOUBLE(FrogYDiffRate, 1.6922887587400708, 1.5, 2.0);PARAM_LOWER(0.0);
	END_TUNING;


	PARAM_DOUBLE(FlogCoinRate, 0.9993396763834262, 0.9, 1.0);PARAM_LOWER(0.0);PARAM_UPPER(1.0);		
	PARAM_DOUBLE(NoDeadProb, 1, 0.0, 1.0);PARAM_LOWER(0.0);PARAM_UPPER(1.0);	


} HP;

constexpr int T = 1000;
constexpr int H_min = 8;
constexpr int H_max = 30;
constexpr int W_min = 8;
constexpr int W_max = 30;
constexpr int WH_max = W_max * H_max;

constexpr int F_min = 1;
constexpr int F_max = 5;

constexpr int KL_min = 2;	
constexpr int KL_max = 6;
constexpr int KW_min = 1;	
constexpr int KW_max = 5;

constexpr double PC_min = 0.2;		
constexpr double PC_max = 0.6;


template <class U> using TArr = CapArr<U, T>;
template <class T> using WHArr = CapArr<T, WH_max>;
template <class T> using WArr = CapArr<T, W_max>;
template <class T> using HArr = CapArr<T, H_max>;
template <class T> using FArr = CapArr<T, F_max>;

template <class T> using WHQue = CapQue<T, WH_max>;



int H;
int W;
int F;
int KL;		
int KW;		
double PC;
HArr<s8> MoveDirs;		

int WH;

namespace CT {
	constexpr char Water = '.';
	constexpr char Log = '=';
	constexpr char Ground = '#';
	constexpr char Frog = '@';
	constexpr char Coin = 'o';

	constexpr char Unknown = '?';
};


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
GridSystemD gs;
AroundMapD<4> aroundMap;
DirMapD<4> dirMap;


struct BeltGrid {
	CapArr<CapArr<char, T + W_max>, H_max> grid_;

	void Init() {
		grid_.resize(H);
		REP(r, H) {
			grid_[r].assign(W + T, CT::Water);
		}
	}

	int ToBc(int t, int r, int c) const {
		if (r == 0) {
			return c;
		}
		else if (MoveDirs[r] < 0) {
			return c + t;
		}
		else {
			return T + c - t;
		}
	}

	int ToC(int t, int r, int bc) const {
		if (r == 0) {
			return bc;
		}
		else if (MoveDirs[r] < 0) {
			return bc - t;
		}
		else {
			return bc - T + t;
		}
	}

	char Get(int t, int r, int c) const {
		return grid_[r][ToBc(t, r, c)];
	}

	void Set(int t, int r, int c, char v) {
		grid_[r][ToBc(t, r, c)] = v;
	}

	bool CanMove(int t, int r, int c) const {
		int bc = ToBc(t, r, c);
		return grid_[r][bc] != CT::Water && grid_[r][bc] != CT::Frog;
	}
};

struct Frog {
	int coin_;		
	int pos_;		

	bool operator == (const Frog& r) const {
		return coin_ == r.coin_ &&
			pos_ == r.pos_;
	}
	bool operator != (const Frog& r) const {
		return !((*this) == r);
	}

	void Kill() {
		pos_ = -1;
	}
	bool IsDead() const {
		return pos_ < 0;
	}
	void Revival(int pos) {
		coin_ = 0;
		pos_ = pos;
	}

};

struct MoveBackup {
	int fi;				
	int pos;			
	char v;				
	int frogCoin;		
	int coin;			
};
struct KillBackup {
	int fi;				
	int pos;			
	int frogCoin;		
};

struct State {
	WHArr<char> grid_;
	FArr<Frog> frogs_;
	int coin_;
	int turn_;

	bool operator == (const State& r) const {
		return grid_ == r.grid_ &&
			frogs_ == r.frogs_ &&
			coin_ == r.coin_ &&
			turn_ == r.turn_;
	}
	bool operator != (const State& r) const {
		return !((*this) == r);
	}

	void Init() {
		grid_.assign(WH, CT::Water);
		frogs_.resize(F);
		REP(fi, F) {
			frogs_[fi].Kill();
		}
		coin_ = 0;
		turn_ = 0;
	}

	bool CanMove(int fi, Dir dir) const {
		cauto& f = frogs_[fi];
		int npos = dirMap[f.pos_][(int)dir];
		pint np = gs.ToPos(npos);

		if (grid_[npos] == CT::Water || grid_[npos] == CT::Frog) {
			return false;
		}
		return true;
	}

	void Move(int fi, Dir dir, MoveBackup& backup) {
		backup.fi = fi;
		backup.coin = coin_;

		auto& f = frogs_[fi];
		backup.frogCoin = f.coin_;

		pint p = gs.ToPos(f.pos_);
		int npos = dirMap[f.pos_][(int)dir];
		pint np = gs.ToPos(npos);

		VASSERT(grid_[npos] != CT::Water);
		VASSERT(grid_[npos] != CT::Frog);

		if (np.y == 0) {
			coin_ += f.coin_;
			f.coin_ = 0;
		}
		else {
			char v = grid_[npos];
			if (v == CT::Coin) {
				f.coin_ += np.y;
			}
		}

		backup.v = grid_[npos];
		backup.pos = f.pos_;

		grid_[npos] = CT::Frog;
		grid_[f.pos_] = p.y == 0 ? CT::Ground : CT::Log;

		f.pos_ = npos;
	}
	void Move(int fi, Dir dir) {
		static MoveBackup backup;
		Move(fi, dir, backup);
	}

	void Restore(const MoveBackup& backup) {
		auto& f = frogs_[backup.fi];
		grid_[f.pos_] = backup.v;
		f.pos_ = backup.pos;
		grid_[f.pos_] = CT::Frog;
		f.coin_ = backup.frogCoin;
		coin_ = backup.coin;
	}

	void Kill(int fi, KillBackup& backup) {
		auto& f = frogs_[fi];
		pint p = gs.ToPos(f.pos_);
		backup.fi = fi;
		backup.frogCoin = f.coin_;
		backup.pos = f.pos_;
		grid_[f.pos_] = p.y == 0 ? CT::Ground : CT::Log;
		f.Kill();
	}

	void Restore(const KillBackup& backup) {
		auto& f = frogs_[backup.fi];
		f.coin_ = backup.frogCoin;
		f.pos_ = backup.pos;
		grid_[f.pos_] = CT::Frog;
	}
};
constexpr int StateSize = sizeof(State);

template <class T, int SimTurn>
struct BeltGrid2 {
	HArr<CapArr<T, W_max + SimTurn>> rows_;

	bool operator == (const BeltGrid2& r) const {
		return rows_ == r.rows_;
	}
	bool operator != (const BeltGrid2& r) const {
		return !((*this) == r);
	}

	void Init(const T& initValue) {
		rows_.resize(H);
		REP(y, H) {
			rows_[y].assign(W + SimTurn, initValue);
		}
	}

	int ToRowX(int t, int y, int x) const {
		if (y == 0) {
			return x;
		}
		else if (MoveDirs[y] < 0) {
			return x + t;
		}
		else {
			return (W - 1 - x) + t;
		}
	}


	const T& GetValue(int t, int y, int x) const {
		return rows_[y][ToRowX(t, y, x)];
	}
	const T& GetValueOpt(int t, int y, int x, const T& defaultValue) const {
		int rowX = ToRowX(t, y, x);
		if (rowX < 0 || rowX >= W + SimTurn) {
			return defaultValue;
		}
		return rows_[y][rowX];
	}
	void SetValue(int t, int y, int x, const T& v) {
		rows_[y][ToRowX(t, y, x)] = v;
	}
};

constexpr int AliveCheckTurn = T + 1;
constexpr int AliveSkipTurn = 200;			

struct AliveProb {
	struct From2 {
		int pos;
		int turn;
	};



	WHArr<TArr<double>> homeProbs;	

	void Init() {
		static BeltGrid2<bool, AliveCheckTurn> movable;
		MakeCase(movable);

		static CapArr<WHArr<From2>, AliveCheckTurn> fromMaps;
		static CapArr<WHArr<int>, AliveCheckTurn> distMaps;
		MakeLiveMap(movable, fromMaps, distMaps);

		static CapArr<WHArr<From2>, AliveCheckTurn> reachFromMaps;
		static CapArr<WHArr<int>, AliveCheckTurn> reachDistMaps;
		MakeReachMap(movable, reachFromMaps, reachDistMaps);

		homeProbs.resize(WH);
		REP(pos, WH) {
			int sampleCount = 0;
			TArr<int> counts;
			counts.assign(T, 0);
			REP(t, AliveCheckTurn - AliveSkipTurn) {
				if (reachDistMaps[t][pos] >= 0) {
					++sampleCount;

					if (distMaps[t][pos] >= 0) {
						FOR(t2, distMaps[t][pos], T) {
							++counts[t2];
						}
					}
				}
			}

			if (sampleCount > 0) {
				homeProbs[pos].resize(T);
				REP(t, T) {
					homeProbs[pos][t] = counts[t] / (double)sampleCount;
				}
			}
			else {
				homeProbs[pos].assign(T, -1);
			}
		}

	}

	double GetHomeProb(int pos, int t) const {
		int turnLeft = (T - t);
		chmin(turnLeft, T - 1);
		double prob = homeProbs[pos][turnLeft];
		return prob;
	}
	double GetHomeProbWithTurnLeft(int pos, int turnLeft) const {
		chmin(turnLeft, T - 1);
		double prob = homeProbs[pos][turnLeft];
		return prob;
	}

	void MakeCase(BeltGrid2<bool, AliveCheckTurn>& movable) {
		Xor64 rand_;

		movable.Init(false);
		REP(r, H) {
			if (r == 0) {
				REP(c, W) {
					movable.rows_[r][c] = true;
				}
			}
			else {
				bool doLog = rand_.GetProb(0.5);
				int c = 0;

				while (true) {
					int length = 0;
					if (doLog) {
						length = 1 + rand_(KL);
					}
					else {
						length = 1 + rand_(KW);
					}

					for (int i = 0; i < length && c < AliveCheckTurn; i++, c++) {
						if (doLog) {
							movable.rows_[r][c] = true;
						}
					}
					doLog = !doLog;
					if (c >= AliveCheckTurn) {
						break;
					}
				}
			}
		}
	}

	void MakeLiveMap(BeltGrid2<bool, AliveCheckTurn>& movable, CapArr<WHArr<From2>, AliveCheckTurn>& fromMaps, CapArr<WHArr<int>, AliveCheckTurn>& distMaps) {

		struct PNode {
			int pos;				
			int turn;				
		};
		queue<PNode> que;

		fromMaps.resize(T + 1);
		distMaps.resize(T + 1);
		REP(t, T + 1) {
			fromMaps[t].assign(WH, { -1, -1 });
			distMaps[t].assign(WH, -1);

			REP(x, W) {
				int pos = gs.ToId(x, 0);
				distMaps[t][pos] = 0;
				que.push(PNode{ pos, t });
			}
		}

		while (!que.empty()) {
			PNode node = que.front();
			que.pop();

			auto Check = [&](const pint& n) {
				int nextTurn = node.turn - 1;
				if (nextTurn < 0) {
					return;
				}
				if (!movable.GetValue(nextTurn, n.y, n.x)) {
					return;
				}
				int npos = gs.ToId(n);
				if (distMaps[nextTurn][npos] >= 0) {
					return;
				}

				int nextDist = distMaps[node.turn][node.pos] + 1;
				fromMaps[nextTurn][npos] = { node.pos, node.turn };
				distMaps[nextTurn][npos] = nextDist;
				que.push(PNode{ npos, nextTurn });
			};


			pint n = gs.ToPos(node.pos);
			n.x -= MoveDirs[n.y];
			if (n.x < 0 || n.x >= W) {
				continue;
			}

			{
				Check(pint{ n.x, n.y });
			}

			if (n.x - 1 >= 0) {
				Check(pint{ n.x - 1, n.y });
			}
			if (n.x + 1 < W) {
				Check(pint{ n.x + 1, n.y });
			}
			if (n.y - 1 >= 0) {
				Check(pint{ n.x, n.y - 1 });
			}
			if (n.y + 1 < H) {
				Check(pint{ n.x, n.y + 1 });
			}
		}
	}

	void MakeReachMap(BeltGrid2<bool, AliveCheckTurn>& movable, CapArr<WHArr<From2>, AliveCheckTurn>& fromMaps, CapArr<WHArr<int>, AliveCheckTurn>& distMaps) {

		struct PNode {
			int pos;				
			int turn;				
		};
		queue<PNode> que;

		fromMaps.resize(T + 1);
		distMaps.resize(T + 1);
		REP(t, T + 1) {
			fromMaps[t].assign(WH, { -1, -1 });
			distMaps[t].assign(WH, -1);

			REP(x, W) {
				int pos = gs.ToId(x, 0);
				distMaps[t][pos] = 0;
				que.push(PNode{ pos, t });
			}
		}

		while (!que.empty()) {
			PNode node = que.front();
			que.pop();

			auto Check = [&](pint n) {
				int nextTurn = node.turn + 1;
				if (nextTurn > T) {
					return;
				}
				if (!movable.GetValue(node.turn, n.y, n.x)) {
					return;
				}

				n.x += MoveDirs[n.y];
				if (n.x < 0 || n.x >= W) {
					return;
				}

				int npos = gs.ToId(n);
				if (distMaps[nextTurn][npos] >= 0) {
					return;
				}

				int nextDist = distMaps[node.turn][node.pos] + 1;
				fromMaps[nextTurn][npos] = { node.pos, node.turn };
				distMaps[nextTurn][npos] = nextDist;
				que.push(PNode{ npos, nextTurn });
			};


			pint n = gs.ToPos(node.pos);
			{
				Check(pint{ n.x, n.y });
			}

			if (n.x - 1 >= 0) {
				Check(pint{ n.x - 1, n.y });
			}
			if (n.x + 1 < W) {
				Check(pint{ n.x + 1, n.y });
			}
			if (n.y - 1 >= 0) {
				Check(pint{ n.x, n.y - 1 });
			}
			if (n.y + 1 < H) {
				Check(pint{ n.x, n.y + 1 });
			}
		}
	}

};

template <class T>
struct ExtGrid {
	CapArr<CapArr<T, W_max + 2>, H_max> grid_;

	void Init(const T value) {
		grid_.resize(H);
		REP(y, H) {
			grid_[y].assign(W + 2, value);
		}
	}

	const T& Get(int y, int x) const {
		return grid_[y][x + 1];
	}
	void Set(int y, int x, const T& v) {
		grid_[y][x + 1] = v;
	}


};


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



bool IsMovable(char v) {
	return v != CT::Water && v != CT::Frog;
}

int SlidedPos(int pos) {
	auto p = gs.ToPos(pos);
	p.x += MoveDirs[p.y];
	if (p.x < 0 || p.x >= W) {
		return -1;
	}
	return gs.ToId(p);
}


int CalcFrontX(int x, int y) {
	if (MoveDirs[max(y, 1)] < 0) {
		return x;
	}
	else {
		return W - 1 - x;
	}
}

int FromFrontX(int frontX, int y) {
	if (MoveDirs[max(y, 1)] < 0) {
		return frontX;
	}
	else {
		return W - 1 - frontX;
	}
}

int ConvertTurn(int curPos, int curTurn, int targetTurn) {
	pint p = gs.ToPos(curPos);
	p.x += MoveDirs[p.y] * (targetTurn - curTurn);
	if (p.x < 0 || p.x >= W) {
		return -1;
	}
	return gs.ToId(p);
}
int ConvertTurn(int x, int y, int curTurn, int targetTurn) {
	x += MoveDirs[y] * (targetTurn - curTurn);
	if (x < 0 || x >= W) {
		return -1;
	}
	return gs.ToId(x, y);
}



struct Command {
	struct Action {
		s8 fi;
		Dir dir;
	};

	FArr<Action> actions;
};

struct IOServer {

	State state_;
	int deadCount_;
	int deadCoin_;

	void InitInput(ChronoTimer& timer) {
		istream& is = cin;
		is >> H >> W >> F >> KL >> KW >> PC;
		timer.Init();

		char dir;
		MoveDirs.assign(H, 0);
		FOR(r, 1, H) {
			is >> dir;
			if (dir == '<') {
				MoveDirs[r] = -1;
			}
			else {
				MoveDirs[r] = 1;
			}
		}

		WH = W * H;
		gs.Init(W, H);
		aroundMap.Init(W, H, Around4);
		dirMap.Init(W, H, Around4);

		state_.Init();
		deadCount_ = 0;
		deadCoin_ = 0;

	}

	void Input(ChronoTimer& timer) {
		istream& is = cin;

		for (auto& f : state_.frogs_) {
			if (f.IsDead()) {
				continue;
			}
			pint p = gs.ToPos(f.pos_);
			int nc = p.x + MoveDirs[p.y];
			if (nc < 0 || nc >= W) {
				++deadCount_;
				deadCoin_ += f.coin_;
				f.Kill();
			}
			else {
				f.pos_ = gs.ToId(nc, p.y);
			}
		}

		char v;
		REP(pos, WH) {
			is >> v;
			state_.grid_[pos] = v;

			if (v == CT::Frog) {
				bool exist = false;
				for (auto& f : state_.frogs_) {
					if (f.IsDead()) {
						continue;
					}
					if (pos == f.pos_) {
						exist = true;
						break;
					}
				}
				if (!exist && gs.ToPos(pos).y == 0) {
					for (auto& f : state_.frogs_) {
						if (f.IsDead()) {
							f.Revival(pos);
							exist = true;
							break;
						}
					}
				}
				VASSERT(exist);
			}
		}


		int elapsedTime;
		is >> elapsedTime;
		timer.SetElapseTimeMs(elapsedTime);


	}

	void Output(const Command& command) {
		ostream& os = cout;
		os << command.actions.size() << endl;

		REP(i, command.actions.size()) {
			int fi = command.actions[i].fi;
			Dir dir = command.actions[i].dir;
			VASSERT(dir != Dir::N);

			auto& f = state_.frogs_[fi];
			pint p = gs.ToPos(f.pos_);

			os << p.y << " " << p.x << " " << DirToChar(dir) << endl;

			state_.Move(fi, dir);
		}

		++state_.turn_;
	}

	void Finalize() {
	}
};
IOServer server;


template <class OBJECT>
struct ObjectPool {
	static_assert(is_trivially_copyable<OBJECT>::value, "not trivially copyable");		

	OBJECT* pool_ = nullptr;			
	OBJECT** reusable_ = nullptr;		
	int capacity_ = 0;			
	int usedTotalCount_ = 0;	
	int usedPoolCount_ = 0;		
	int reusableCount_ = 0;		

	int maxTotalCount_ = 0;

	~ObjectPool() {
		if (pool_) {
			free(pool_);
		}
		if (reusable_) {
			free(reusable_);
		}
	}


	void Init(int capacity) {
		if (capacity != capacity_) {
			if (pool_) {
				free(pool_);
			}
			if (reusable_) {
				free(reusable_);
			}
			pool_ = (OBJECT*)malloc(capacity * sizeof(OBJECT));
			reusable_ = (OBJECT**)malloc(capacity * sizeof(OBJECT*));
			capacity_ = capacity;
		}
		usedTotalCount_ = 0;
		usedPoolCount_ = 0;
		reusableCount_ = 0;
	}

	void Clear() {
		usedTotalCount_ = 0;
		usedPoolCount_ = 0;
		reusableCount_ = 0;
	}

	inline OBJECT* New() {
		if (reusableCount_) {
			++usedTotalCount_;
			--reusableCount_;
			chmax(maxTotalCount_, usedTotalCount_);
			return reusable_[reusableCount_];
		}

		if (usedPoolCount_ >= capacity_) {
		}
		++usedTotalCount_;
		chmax(maxTotalCount_, usedTotalCount_);
		return &pool_[usedPoolCount_++];
	}

	inline void Delete(OBJECT* obj) {
		reusable_[reusableCount_] = obj;
		++reusableCount_;
		--usedTotalCount_;
	}

	inline void Delete(OBJECT** start, int count) {
		memcpy(&reusable_[reusableCount_], start, sizeof(OBJECT*) * count);
		reusableCount_ += count;
		usedTotalCount_ -= count;
	}

	inline int GetUsedCount() const {
		return usedTotalCount_;
	}

	inline string Usage() const {
		stringstream ss;
		ss << usedTotalCount_ << " (" << (int)floor(usedTotalCount_ * 100 / (double)capacity_ + 0.5) << " %)";
		return ss.str();
	}

	inline int UsedRate() const {
		return (int)round(usedTotalCount_ * 100 / (double)capacity_);
	}

	inline int GetCapacity() const {
		return capacity_;
	}

	inline int GetMaxUseCount() const {
		return maxTotalCount_;
	}

	inline int GetMaxUsedRate() const {
		return (int)round((double)GetMaxUseCount() * 100 / (double)capacity_);
	}
	inline string MaxUsage() const {
		stringstream ss;
		ss << maxTotalCount_ << " / " << capacity_ << " (" << (int)floor(maxTotalCount_ * 100 / (double)capacity_ + 0.5) << " %)";
		return ss.str();
	}
	double GetMemoryRate() const {
		return usedTotalCount_ / (double)capacity_;
	}
};

inline int popcount(uint64_t bit) {
#if _MSC_VER
	return (int)__popcnt64(bit);
#else
	return __builtin_popcountll(bit);
#endif
}

int msb(uint64_t v) {
	if (v == 0) return false;
	v |= (v >> 1);
	v |= (v >> 2);
	v |= (v >> 4);
	v |= (v >> 8);
	v |= (v >> 16);
	v |= (v >> 32);
	return popcount(v) - 1;
}

inline uint64_t RotateLeft(uint64_t bit, uint64_t shift) {
	return (bit << shift) | (bit >> (64ULL - shift));
}
inline uint64_t RotateRight(uint64_t bit, uint64_t shift) {
	return (bit >> shift) | (bit << (64ULL - shift));
}


#if _MSC_VER
#include <intrin.h>
#endif

inline int FindBitRL(uint64_t bit) {
	if (bit == 0) {
		return -1;
	}
#if _MSC_VER
	unsigned long index = 0;
	_BitScanForward64(&index, bit);
	return (int)index;
#else
	return __builtin_ctzll(bit);
#endif
}

inline int FindBitLR(uint64_t bit) {
	if (bit == 0) {
		return -1;
	}
#if _MSC_VER
	unsigned long index = 0;
	_BitScanReverse64(&index, bit);
	return 63 - (int)index;			
#else
	return __builtin_clzll(bit);
#endif
}



template <class K>
struct CapHashSet {
	size_t m_capacity;
	K* m_keys;
	uint8_t* m_states;
	size_t m_mask;
	int m_count = 0;


	void init(size_t capacity) {
		m_capacity = 1ULL << (msb(capacity) + 1);
		m_mask = m_capacity - 1;
		m_keys = (K*)malloc(m_capacity * sizeof(K));
		m_states = (uint8_t*)calloc(m_capacity, sizeof(uint8_t));
		m_count = 0;
	}

	void clear() {
		memset(m_states, 0, m_capacity * sizeof(uint8_t));
		m_count = 0;
	}

	inline void set(K key) {
		size_t i = key & m_mask;
		for (;;) {
			if (m_states[i]) {
				if (key == m_keys[i]) {
					break;
				}
			}
			else {
				m_states[i] = 1;
				m_keys[i] = key;
				++m_count;
				break;
			}
			(++i) &= m_mask;
		}
	}
	inline void insert(K key) {
		set(key);
	}

	inline bool contains(K key) const {
		size_t i = key & m_mask;
		for (;;) {
			if (m_states[i]) {
				if (key == m_keys[i]) {
					return true;
				}
			}
			else {
				break;
			}
			(++i) &= m_mask;
		}
		return false;
	}

	bool enter(K key) {
		size_t i = key & m_mask;
		for (;;) {
			if (m_states[i]) {
				if (key == m_keys[i]) {
					return false;
				}
			}
			else {
				m_states[i] = 1;
				m_keys[i] = key;
				++m_count;
				return true;
			}
			(++i) &= m_mask;
		}
	}

	int GetCount() const {
		return m_count;
	}

};

#define USE_HISTOGRAM 1
#define USE_HISTOGRAM_RANGE 0	


struct NthHistogram {
	double minValue_ = 0;
	double maxValue_ = 0;
	double valueRange_ = 0;
	int resolution_ = 0;
	vector<int> histogram_;
	int border_ = 0;				
	double borderOverProb_ = 0;		

	void Init(double minValue, double maxValue, int resolution) {
		minValue_ = minValue;
		maxValue_ = maxValue;
		valueRange_ = maxValue - minValue;
		resolution_ = resolution;
		histogram_.assign(resolution, 0);
		border_ = -1;
		borderOverProb_ = 0;
	}

	int CalcHi(double value) const {
		if (valueRange_ == 0) {
			return 0;
		}
		int hi = (int)(resolution_ * (value - minValue_) / valueRange_);
		VASSERT(hi >= 0);
		chmin(hi, (int)resolution_ - 1);
		return hi;
	}

	void Vote(double value) {
		++histogram_[CalcHi(value)];
	}

	void UpdateBorder(int n) {
		border_ = -1;
		borderOverProb_ = 0;

		int cnt = 0;
		RREP(hi, histogram_.size()) {
			int nextCnt = cnt + histogram_[hi];
			if (nextCnt > n) {
				border_ = hi;
				borderOverProb_ = (n - cnt) / (double)histogram_[hi];
				break;
			}
			cnt = nextCnt;
		}
	}

	bool IsOver(double value, Xor64& rand) const {
		int hi = CalcHi(value);
		if (hi > border_) {
			return true;
		}
		if (hi == border_&& rand.GetProb(borderOverProb_)) {
			return true;
		}
		return false;
	}

};

struct NthRange {
	int width_ = 0;
	int count_ = 0;
	double minValue_ = 0;
	double maxValue_ = 0;

	void init(int width) {
		width_ = width;
		count_ = 0;
		minValue_ = DBL_MAX;
		maxValue_ = -DBL_MAX;
	}

	void clear() {
		count_ = 0;
		minValue_ = DBL_MAX;
		maxValue_ = -DBL_MAX;
	}

	bool TryPush(double value) {
		if (CanPush(value)) {
			Push(value);
			return true;
		}
		return false;
	}

	bool CanPush(double value) {
		if (count_ < width_) {
			return true;
		}
		return value >= minValue_;
	}

	void Push(double value) {
		chmin(minValue_, value);
		chmax(maxValue_, value);
	}

};

#define MULTI_BEAM_LOG 0
#define SINGLE_BEAM_LOG 0


template <class NODE, class STATE, class BACKUP>
struct DiffBeamHash {
public:
	struct NodeEx : public NODE {
		NodeEx* parent;			
		NodeEx* child;			
		NodeEx* prev;			
		NodeEx* next;			
		double evalScore;
	};

private:
	using NodePool = ObjectPool<NodeEx>;
	using HashTable = CapHashSet<u64>;

	NodePool nodePool_;
	vector<NodeEx*> targets_;
	vector<NodeEx*> dels_;
	HashTable hashTbl_;

public:
	int makeNextCount_ = 0;


public:
	NODE* New() {
		NodeEx* node = nodePool_.New();
		return node;
	}
	void Delete(NODE* node) {
		nodePool_.Delete((NodeEx*)node);
	}

	void GetBest(const NODE* bestNode, vector<const NODE*>& nodes) {
		nodes.clear();
		const NodeEx* node = (NodeEx*)bestNode;
		while (node->parent) {
			nodes.emplace_back(node);
			node = node->parent;
		}
		reverse(ALL(nodes));
	}

	void ReleaseNode(NodeEx* node) {
		VASSERT(node->child == nullptr);

		while (true) {
			NodeEx* next = nullptr;
			if (node->parent) {
				if (node->parent->child == node) {
					VASSERT(node->prev == nullptr);
					if (node->next) {
						node->parent->child = node->next;
						node->next->prev = nullptr;
					}
					else {
						node->parent->child = nullptr;
						next = node->parent;
					}
				}
				else {
					NodeEx* prev = node->prev;
					NodeEx* next = node->next;
					VASSERT(prev);
					prev->next = next;
					if (next) {
						next->prev = prev;
					}
				}
			}
			nodePool_.Delete(node);

			if (next) {
				node = next;
			}
			else {
				break;
			}
		}
	}

	void Initialize(int widthLimit, int maxExpandCount, int turnMax) {
		targets_.reserve(widthLimit * maxExpandCount);
		dels_.reserve(widthLimit * maxExpandCount);
		nodePool_.Init(widthLimit * turnMax * 2);
		makeNextCount_ = 0;
		hashTbl_.init(widthLimit * maxExpandCount * 4);
	}

	template <class INIT_ROOT, class MAKE_NEXT_NODES, class UPDATE_STATE, class REVERT_STATE, class CALC_WIDTH, class IS_END>
	void Run(int turnMax, INIT_ROOT&& initRoot, MAKE_NEXT_NODES&& makeNextNodes, UPDATE_STATE&& updateState, REVERT_STATE&& revertState, CALC_WIDTH&& calcWidth, IS_END&& isEnd, int realTurn) {
		targets_.clear();
		dels_.clear();
		nodePool_.Clear();
		makeNextCount_ = 0;
		hashTbl_.clear();

		STATE state;

		NodeEx* rootNode = nodePool_.New();
		{
			rootNode->parent = nullptr;
			rootNode->child = nullptr;
			rootNode->prev = nullptr;
			rootNode->next = nullptr;
			rootNode->evalScore = 0;
		}

		initRoot(state, rootNode);

		int startDepth = 0;
		NodeEx* startNode = rootNode;

		REP(turn, turnMax) {
			int targetWidth = calcWidth(turn, makeNextCount_);


			auto dfs = [&](auto& dfs, int depth, NodeEx* node) -> void {
				if (depth >= turn) {
					static vector<pr<NODE*, double>> nexts;
					nexts.clear();
					makeNextNodes(depth, node, state, nexts, hashTbl_);

					++makeNextCount_;

					NodeEx* prev = nullptr;
					for (auto&& [nextTmp, evalScore] : nexts) {
						NodeEx* next = (NodeEx*)nextTmp;
						if (node->child == nullptr) {
							node->child = next;
							next->parent = node;
							next->child = nullptr;
							next->prev = nullptr;
							next->next = nullptr;
						}
						else {
							prev->next = next;
							next->parent = node;
							next->child = nullptr;
							next->prev = prev;
							next->next = nullptr;
						}
						next->evalScore = evalScore;
						prev = next;
					}

					if (node->child) {
						NodeEx* child = node->child;
						while (child) {
							targets_.emplace_back(child);
							child = child->next;
						}
					}
					else {
						dels_.emplace_back(node);
					}
				}
				else {
					BACKUP backup;
					NodeEx* child = node->child;
					while (child) {
						updateState(state, child, backup);
						dfs(dfs, depth + 1, child);
						revertState(state, child, backup);
						child = child->next;
					}
				}
			};

			hashTbl_.clear();

			dfs(dfs, startDepth, startNode);


			if (isEnd()) {
				break;
			}

			if (targets_.size() > targetWidth) {
				static NthHistogram hist;
				static Xor64 rand;
				double minValue = 1e100;
				double maxValue = -1e100;
				for (NodeEx* t : targets_) {
					chmin(minValue, t->evalScore);
					chmax(maxValue, t->evalScore);
				}
				hist.Init(minValue, maxValue, 100);
				for (NodeEx* t : targets_) {
					hist.Vote(t->evalScore);
				}
				hist.UpdateBorder(targetWidth);

				for (NodeEx* t : targets_) {
					if (!hist.IsOver(t->evalScore, rand)) {
						ReleaseNode(t);
					}
				}

			}
			else {
			}

			for (NodeEx* d : dels_) {
				ReleaseNode(d);
			}
			dels_.clear();
			targets_.clear();

			BACKUP backup;
			while (startNode->child && startNode->child->next == nullptr) {
				updateState(state, startNode->child, backup);
				startNode = startNode->child;
				++startDepth;
			}
		}


	}
};

struct SimFrogMoveBackup {

};

struct SimFrog {
	int coin_;				
	int pos_;				
	int deadTurn_;			
	bool noDeadMode_;		
	bool onNoDeadLog_;		

	bool operator == (const SimFrog& r) const {
		return coin_ == r.coin_ &&
			pos_ == r.pos_ &&
			deadTurn_ == r.deadTurn_ &&
			noDeadMode_ == r.noDeadMode_ &&
			onNoDeadLog_ == r.onNoDeadLog_;
	}
	bool operator != (const SimFrog& r) const {
		return !((*this) == r);
	}

	void SetFrog(const Frog& frog) {
		coin_ = frog.coin_;
		pos_ = frog.pos_;
		deadTurn_ = -1;
		noDeadMode_ = false;
		onNoDeadLog_ = false;
	}

	void Kill(int turn) {
		deadTurn_ = turn;
	}
	bool IsDead() const {
		return deadTurn_ >= 0;
	}
	void RevertKill() {
		deadTurn_ = -1;
	}


};

struct SimMoveBackup {
	int fi;				
	int pos;			
	pint from;			
	pint to;			
	char v;				
	int frogCoin;		
	int coin;			
	bool noDeadMode_;	
	bool onNoDeadLog_;	
};
struct SimKillBackup {
	int fi;				
};

struct SlideBackup {
	FArr<KillBackup> killBackups;
};
struct SimSlideBackup {
	FArr<SimKillBackup> killBackups;
};

constexpr int SimTurn = 1;

struct SimState {
	BeltGrid2<char, SimTurn> grid_;

	FArr<SimFrog> frogs_;
	int coin_;
	int turn_;

	int offsetTurn_;

	bool operator == (const SimState& r) const {
		return
			grid_ == r.grid_ &&
			frogs_ == r.frogs_ &&
			coin_ == r.coin_ &&
			turn_ == r.turn_ &&
			offsetTurn_ == r.offsetTurn_;
	}
	bool operator != (const SimState& r) const {
		return !((*this) == r);
	}

	void SetState(const State& state, const WHArr<s8>& lifeMap) {
		frogs_.resize(F);
		REP(fi, F) {
			auto& f = frogs_[fi];
			f.SetFrog(state.frogs_[fi]);

			if (lifeMap[f.pos_] == 1) {
				f.onNoDeadLog_ = true;
			}
		}
		coin_ = state.coin_;
		turn_ = state.turn_;

		grid_.Init(CT::Water);
		REP(y, H) {
			REP(x, W) {
				char v = state.grid_[gs.ToId(x, y)];
				grid_.SetValue(0, y, x, v);
			}
		}

		FOR(y, 1, H) {
			auto& row = grid_.rows_[y];

			bool nextLog = true;
			FOR(x, W - KW, W) {
				if (row[x] != CT::Water) {
					nextLog = false;
					break;
				}
			}
			if (nextLog) {
				row[W] = CT::Log;
			}
		}

		offsetTurn_ = 0;
	}




	char GetSimValue(int y, int x) const {
		return grid_.GetValueOpt(offsetTurn_, y, x, CT::Water);
	}


	char GetValue(int pos) const {
		pint p = gs.ToPos(pos);
		return grid_.GetValueOpt(offsetTurn_, p.y, p.x, CT::Water);
	}

	void SetValue(int pos, char v) {
		cauto& p = gs.ToPos(pos);
		grid_.SetValue(offsetTurn_, p.y, p.x, v);
	}

	bool CanMove(int fi, Dir dir) const {
		if (dir == Dir::N) {
			return true;
		}
		cauto& f = frogs_[fi];
		int npos = dirMap[f.pos_][(int)dir];
		pint np = gs.ToPos(npos);

		char nv = GetValue(npos);
		if (nv == CT::Water || nv == CT::Frog) {
			return false;
		}
		return true;
	}








	void Move(int fi, Dir dir, const WHArr<s8>& lifeMap, SimMoveBackup& backup, bool overFrog, bool underFrog) {
		if (dir == Dir::N) {
			backup.fi = -1;
			return;
		}
		auto& f = frogs_[fi];

		const pint p = gs.ToPos(f.pos_);
		const int npos = dirMap[f.pos_][(int)dir];
		const pint n = gs.ToPos(npos);
		int fromRowX = grid_.ToRowX(offsetTurn_, p.y, p.x);
		int toRowX = grid_.ToRowX(offsetTurn_, n.y, n.x);

		backup.fi = fi;
		backup.coin = coin_;
		backup.frogCoin = f.coin_;
		backup.pos = f.pos_;
		backup.from.x = fromRowX;
		backup.from.y = p.y;
		backup.to.x = toRowX;
		backup.to.y = n.y;
		backup.noDeadMode_ = f.noDeadMode_;
		backup.onNoDeadLog_ = f.onNoDeadLog_;

		VASSERT(grid_.rows_[n.y][toRowX] != CT::Water);
		const char v = grid_.rows_[n.y][toRowX];
		backup.v = v;

		if (n.y == 0) {
			coin_ += f.coin_;
			f.coin_ = 0;
		}
		else {
			if (v == CT::Coin) {
				f.coin_ += n.y;
			}
		}

		int zeroNpos = ConvertTurn(n.x, n.y, turn_, server.state_.turn_);

		if (f.noDeadMode_) {
			f.noDeadMode_ = false;
		}
		else {
			int zeroPos = ConvertTurn(p.x, p.y, turn_, server.state_.turn_);
			if (zeroNpos >= 0 && zeroPos >= 0) {
				if (f.onNoDeadLog_) {
					if (p.y > 0 && (dir == Dir::L || dir == Dir::R)) {
						if (CalcFrontX(p.x, p.y) == 0 && CalcFrontX(n.x, n.y) == 1) {
							if (lifeMap[zeroPos] == 1 && lifeMap[zeroNpos] == 2) {
								f.noDeadMode_ = true;
							}
						}
					}
				}
			}
		}

		if (f.onNoDeadLog_) {
			if (IsHorizontal(dir) && zeroNpos >= 0 && lifeMap[zeroNpos] == 1) {
			}
			else {
				f.onNoDeadLog_ = false;
			}
		}

		if (!f.onNoDeadLog_) {
			if (n.y > 0) {
				int elapsedTurn = turn_ - server.state_.turn_;
				int limitFrontX = 1 + elapsedTurn * 2;		
				int frontX = CalcFrontX(n.x, n.y);
				if (frontX >= limitFrontX && zeroNpos >= 0 && lifeMap[zeroNpos] == 1) {
					f.onNoDeadLog_ = true;
				}
			}
		}
		if (!f.onNoDeadLog_) {
			if (n.y <= 1) {
				f.onNoDeadLog_ = true;
			}
			else {
				bool hasBackDir = false;
				if (MoveDirs[n.y] != MoveDirs[n.y - 1]) {
					hasBackDir = true;
				}
				else if (n.y + 1 < H && MoveDirs[n.y] != MoveDirs[n.y + 1]) {
					hasBackDir = true;
				}
				if (hasBackDir) {
					int moveDir = MoveDirs[n.y];
					int logLen = 1;
					int frontX = CalcFrontX(n.x, n.y) - 1;		
					if (frontX >= 0) {
						int frontMargin = 0;		
						bool hasBack = false;
						if (MoveDirs[n.y] < 0) {
							RFOR(x, 1, n.x) {
								if (GetSimValue(n.y, x) == CT::Water) {
									break;
								}
								++logLen;
							}
							frontMargin = CalcFrontX(n.x, n.y) - logLen;

							FOR(x, n.x + 1, W) {
								if (GetSimValue(n.y, x) == CT::Water) {
									break;
								}
								++logLen;
								hasBack = true;
							}
						}
						else {
							FOR(x, n.x + 1, W - 1) {
								if (GetSimValue(n.y, x) == CT::Water) {
									break;
								}
								++logLen;
							}
							frontMargin = CalcFrontX(n.x, n.y) - logLen;

							RREP(x, n.x) {
								if (GetSimValue(n.y, x) == CT::Water) {
									break;
								}
								++logLen;
								hasBack = true;
							}
						}

						if (logLen >= 2) {
							int logX = -(frontX - KW);
							if (!hasBack) {
								logX += 2;
							}

							int liveTurn = frontMargin + logLen - 1;
							if (liveTurn >= logX) {
								f.onNoDeadLog_ = true;
							}
						}
					}
				}
			}
		}

		grid_.rows_[n.y][toRowX] = CT::Frog;
		if (underFrog) {
			VASSERT(grid_.rows_[p.y][fromRowX] == CT::Frog);
		}
		else {
			grid_.rows_[p.y][fromRowX] = (p.y == 0 ? CT::Ground : CT::Log);
		}
		f.pos_ = npos;

	}

	void Restore(const SimMoveBackup& backup) {
		if (backup.fi < 0) {
			return;
		}
		auto& f = frogs_[backup.fi];
		grid_.rows_[backup.to.y][backup.to.x] = backup.v;
		f.pos_ = backup.pos;
		grid_.rows_[backup.from.y][backup.from.x] = CT::Frog;
		f.coin_ = backup.frogCoin;
		coin_ = backup.coin;
		f.noDeadMode_ = backup.noDeadMode_;
		f.onNoDeadLog_ = backup.onNoDeadLog_;
	}



	void Kill(int fi, SimKillBackup& backup) {
		auto& f = frogs_[fi];
		pint p = gs.ToPos(f.pos_);
		backup.fi = fi;
		SetValue(f.pos_, p.y == 0 ? CT::Ground : CT::Log);
		f.Kill(turn_);

	}
	void Restore(const SimKillBackup& backup) {
		auto& f = frogs_[backup.fi];
		f.RevertKill();
		SetValue(f.pos_, CT::Frog);
	}

	void Slide(SimSlideBackup& backup) {
		backup.killBackups.clear();

		REP(fi, F) {
			auto& f = frogs_[fi];
			if (f.IsDead()) {
				continue;
			}

			pint p = gs.ToPos(f.pos_);
			int nc = p.x + MoveDirs[p.y];
			if (nc < 0 || nc >= W) {
				Kill(fi, backup.killBackups.push());
			}
			else {
				f.pos_ = gs.ToId(nc, p.y);
			}
		}
		++offsetTurn_;
		++turn_;
	}
	void Restore(const SimSlideBackup& backup) {
		--turn_;
		--offsetTurn_;

		REP(fi, F) {
			auto& f = frogs_[fi];
			if (f.IsDead()) {
				continue;
			}

			pint p = gs.ToPos(f.pos_);
			int nc = p.x - MoveDirs[p.y];
			VASSERT(nc >= 0 && nc < W) {
				f.pos_ = gs.ToId(nc, p.y);
			}
		}

		for (cauto& killBackup : backup.killBackups) {
			Restore(killBackup);
		}
	}

};
constexpr int SimStateSize = sizeof(State);


struct ZobristHashD {
private:
	vector<uint64_t> hashMap_;
	int place_;
	int kind_;

public:
	inline void Init(int place, int kind, Xor64& rand) {
		place_ = place;
		kind_ = kind;

		hashMap_.resize(place_ * kind_);
		for (uint64_t& v : hashMap_) {
			v = rand();
		}
	}

	inline uint64_t Hash(int i, int k) const {
		return hashMap_[k + i * kind_];
	}
};


constexpr int ExpandCountMax = 5;
constexpr int TurnMax = T;
constexpr int MaxBeamWidth = 2000;

struct Backup {
	SimMoveBackup moveBackup;
	SimSlideBackup slideBackup;
};

struct Node {
	int fi;						
	Dir dir;					
	double baseScore;			
	FArr<double> frogScores;	
	FArr<u8> frogDirBits;		
	int overFi;					
	bool underFlog;				
	double totalScore;			
	u64 hash;					
};
constexpr int NodeSize = sizeof(Node);

struct NodeCompare {
	bool operator()(const Node* a, const Node* b) const {
		return a->totalScore < b->totalScore;
	}
};

struct Solver4 {
	Xor64 rand_;
	AliveProb aliveProb_;
	int homeStartTurn_;
	int outEndTurn_;
	ZobristHashD zob_;

	WHArr<s8> lifeMap_;			
	ExtGrid<s8> homeYMap_;		
	ExtGrid<s8> outYMap_;		

	DiffBeamHash<Node, SimState, Backup> beam_;
	int expandCount_ = 0;
	int startTime_ = 0;

	void Run(ChronoTimer& timer) {
		Setup();

		REP(t, T) {
			server.Input(timer);
			if (t == 0) {
				startTime_ = timer.ElapseTimeUs();
			}

			MakeLifeMap(server.state_, lifeMap_);

			SimState state;
			state.SetState(server.state_, lifeMap_);

			MakeHomeYMap2(state, homeYMap_);
			MakeOutYMap2(state, outYMap_);

			Command command;
			Beam(timer, state, command);

			server.Output(command);
		}
	}

	void Setup() {
		zob_.Init(H * (W+2), 3, rand_);	
		aliveProb_.Init();

		beam_.Initialize(MaxBeamWidth, ExpandCountMax, TurnMax);

		homeStartTurn_ = 900;

		int bestPos = -1;
		double bestProb = 1000;
		RREP(y, H) {
			REP(x, W) {
				int pos = gs.ToId(x, y);
				double prob = aliveProb_.homeProbs[pos][T - 1];
				if (prob < 0.5) {
					continue;
				}
				if (prob == 1) {
					prob -= abs(x - W / 2);
				}
				if (prob < bestProb) {
					bestProb = prob;
					bestPos = pos;
				}
			}
			if (bestPos >= 0) {
				break;
			}
		}
		cauto& probs = aliveProb_.homeProbs[bestPos];
		double homeTurn = 100;
		RREP(t, T - 1) {
			if (probs[t] != probs[t + 1]) {
				homeTurn = t;
				break;
			}
		}


		{
			double appendTurn = 0;

			int y = 1;
			while (y < H) {
				int len = 1;
				FOR(add, 1, 3) {
					if (y + add >= H) {
						break;
					}
					if (MoveDirs[y + add] == MoveDirs[y]) {
						++len;
					}
					else {
						break;
					}
				}

				if (len == 1) {
				}
				else if (len == 2) {
					appendTurn += HP.SameDir2Turn;
				}
				else {
					VASSERT(len == 3);
					appendTurn += HP.SameDir3Turn;
				}

				y += len;
			}

			homeTurn += appendTurn;
		}

		{
			double LogExpectLength = (KL + 1) / 2.0;
			double WaterExpectLength = (KW + 1) / 2.0;
			double waterRate = WaterExpectLength / (LogExpectLength + WaterExpectLength);

			homeTurn += waterRate * HP.WaterRateHomeTurn;
		}

		homeStartTurn_ = T - int(round(homeTurn * HP.HomeTurnRate));

		outEndTurn_ = int(round(homeTurn * HP.OutTurnRate));


	}

	void Beam(ChronoTimer& timer, const SimState& baseState, Command& command) {
		Node* bestNode = nullptr;
		int bestTurn = -1;

		int TurnDepth = int(round(HP.DepthGeta + W * HP.DepthRate));
		chmax(TurnDepth, 1);
		int TargetDepth = TurnDepth * F;		
		int leftTurn = T - server.state_.turn_;
		chmin(TargetDepth, leftTurn * F);		

		int curTime = timer.ElapseTimeUs();
		int elapsedTime = curTime - startTime_;
		int leftTime = TIME_LIMIT_US - curTime;
		double turnTime = leftTime / (double)leftTurn;				
		timer.StartUs(curTime + int(turnTime));

		int beamWidth;
		if (expandCount_ < 100) {
			beamWidth = 100;
		}
		else {
			double expandTime = elapsedTime / (double)expandCount_;		
			double depthTime = turnTime / (double)TargetDepth;			
			beamWidth = int(depthTime / expandTime);
			chmax(beamWidth, 100);
			chmin(beamWidth, MaxBeamWidth);
		}
		chmin(beamWidth, MaxBeamWidth);

		auto makeRoot = [&](SimState& state, Node* node) {
			state = baseState;
			node->fi = -1;
			node->baseScore = CalcBaseScore(state);
			node->totalScore = node->baseScore;
			node->frogScores.resize(F);
			node->frogDirBits.assign(F, 0);
			REP(fi, F) {
				node->frogScores[fi] = CalcFrogScore(state, fi);
				node->totalScore += node->frogScores[fi];
			}
			node->overFi = -1;
			node->underFlog = false;
			node->hash = 0;
			REP(y, H) {
				REP(x, W) {
					char v = state.GetValue(gs.ToId(x, y));
					node->hash ^= GetCellHash(0, y, x, v);
				}
			}
		};

		auto updateState = [&](SimState& state, const Node* nextNode, Backup& backup) {
			state.Move(nextNode->fi, nextNode->dir, lifeMap_, backup.moveBackup, nextNode->overFi >= 0, nextNode->underFlog);
			if (nextNode->fi == F - 1) {
				state.Slide(backup.slideBackup);
			}
		};

		auto revertState = [&](SimState& state, const Node* nextNode, const Backup& backup) {
			if (nextNode->fi == F - 1) {
				state.Restore(backup.slideBackup);
			}
			state.Restore(backup.moveBackup);
		};

		auto makeNextNodes = [&](int depth, const Node* node, SimState& state, vector<pr<Node*, double>>& nexts, CapHashSet<u64>& hashTbl) {
			int fi = node->fi + 1;
			if (fi >= F) {
				fi = 0;
			}
			cauto& frog = state.frogs_[fi];
			const u8 dirBit = node->frogDirBits[fi];


			auto Add = [&](int di, bool canDead) -> bool {

				Dir dir = Dir(di);

				if (dirBit != 0) {
					if ((dirBit >> di) & 1) {
					}
					else {
						return false;
					}
				}

				int npos = frog.pos_;
				if (dir != Dir::N) {
					npos = dirMap[frog.pos_][di];
					if (npos < 0) {
						return false;
					}
				}

				char nv = state.GetValue(npos);

				int overFi = -1;
				if (dir != Dir::N) {
					if (nv == CT::Water) {
						return false;
					}
					if (nv == CT::Frog) {
						REP(fi2, F) {
							if (fi2 == fi) {
								continue;
							}
							cauto& f2 = state.frogs_[fi2];
							if (f2.pos_ == npos) {
								if (fi2 > fi) {
									overFi = fi2;
								}
								break;
							}
						}
						if (overFi < 0) {
							return false;
						}
					}
				}

				if (!canDead) {
					cauto& n = gs.ToPos(npos);
					if (n.y != 0 && CalcFrontX(n.x, n.y) == 0) {
						return false;
					}
				}

				double newBaseScore = 0;
				double newFrogScore = 0;
				u64 newHash = node->hash;
				{
					int hashTurn = 0;
					int hashPos1 = 0;
					int hashPos2 = 0;
					if (dir != Dir::N) {
						hashTurn = state.turn_ - server.state_.turn_;
						hashPos1 = frog.pos_;
						hashPos2 = dirMap[frog.pos_][(int)dir];
						VASSERT(hashPos2 >= 0);
						newHash ^= GetCellHash(hashTurn, hashPos1, CT::Frog);
						newHash ^= GetCellHash(hashTurn, hashPos2, nv);
					}

					static Backup backup;
					state.Move(fi, dir, lifeMap_, backup.moveBackup, overFi >= 0, dirBit != 0);

					if (dir != Dir::N) {
						newHash ^= GetCellHash(hashTurn, hashPos1, (dirBit != 0 ? CT::Frog : CT::Log));		
						newHash ^= GetCellHash(hashTurn, hashPos2, CT::Frog);		
					}

					state.Slide(backup.slideBackup);

					newBaseScore = CalcBaseScore(state);
					newFrogScore = CalcFrogScore(state, fi);

					state.Restore(backup.slideBackup);
					state.Restore(backup.moveBackup);
				}

				if (!hashTbl.enter(newHash)) {
					return true;
				}

				Node* nextNode = beam_.New();
				nextNode->fi = fi;
				nextNode->dir = dir;
				nextNode->baseScore = newBaseScore;
				nextNode->totalScore = newBaseScore;
				nextNode->frogScores.MemCopy(node->frogScores);
				nextNode->frogScores[fi] = newFrogScore;	
				for (double fs : nextNode->frogScores) {
					nextNode->totalScore += fs;
				}
				nextNode->frogDirBits = node->frogDirBits;
				nextNode->frogDirBits[fi] = 0;
				nextNode->overFi = overFi;
				nextNode->underFlog = (dirBit != 0);

				if (overFi >= 0) {
					Dir back = Back(dir);
					u8 bit = 0;
					REP(di, 4) {
						if (Dir(di) == back) {
							continue;
						}
						bit |= (1 << di);
					}
					nextNode->frogDirBits[overFi] = bit;
				}

				if (state.turn_ + 1 >= T) {
					nextNode->totalScore = nextNode->baseScore;		
				}

				nextNode->hash = newHash;

				nexts.emplace_back();
				auto& next = nexts.back();
				next.first = nextNode;
				next.second = nextNode->totalScore;

				if (depth == TargetDepth - 1 && (bestNode == nullptr || nextNode->totalScore > bestNode->totalScore)) {
					bestNode = nextNode;
					bestTurn = state.turn_ + 1;
				}

				return true;
			};

			if (frog.IsDead()) {
				Add((int)Dir::N, true);
			}
			else {
				bool moved = false;
				REP(di, 5) {
					if (Add(di, false)) {
						moved = true;
					}
				}
				if (!moved) {
					Add((int)Dir::N, true);
				}
			}
		};

		auto calcWidth = [&](int depth, int totalWidth) {
			return beamWidth;
		};

		auto isEnd = [&]() {
			return false;
		};

		beam_.Run(TargetDepth, makeRoot, makeNextNodes, updateState, revertState, calcWidth, isEnd, server.state_.turn_);

		expandCount_ += beam_.makeNextCount_;

		if (bestNode) {

			static vector<const Node*> route;
			route.reserve(TargetDepth);
			beam_.GetBest(bestNode, route);

			FArr<Command::Action> actions;
			REP(i, min(F, (int)route.size())) {
				cauto& node = route[i];
				VASSERT(i == node->fi);
				if (node->dir != Dir::N) {
					auto& action = actions.push();
					action.fi = node->fi;
					action.dir = node->dir;
				}
			}

			auto state = server.state_;
			REP(loop, F) {
				bool updated = false;
				REP(i, actions.size()) {
					auto& a = actions[i];
					if (a.fi < 0) {
						continue;
					}
					if (state.CanMove(a.fi, a.dir)) {
						state.Move(a.fi, a.dir);
						command.actions.push(a);
						a.fi = -1;
						updated = true;
					}
				}
				if (!updated) {
					break;
				}
			}
		}
		else {
		}
	}

	double CalcBaseScore(const SimState& state) const {
		double score = 0;
		score += state.coin_;
		return score;
	}

	double CalcFrogScore(const SimState& state, int fi) const {
		if (state.turn_ >= T) {
			return 0;
		}

		static double LogExpectLength = (KL + 1) / 2.0;
		static double WaterExpectLength = (KW + 1) / 2.0;
		static double LogRate = LogExpectLength / (double)(LogExpectLength + WaterExpectLength);
		static double CoinRate = LogRate * PC;
		static double ScoreRate = H * F * CoinRate;		


		double score = 0;

		cauto& f = state.frogs_[fi];
		if (f.IsDead()) {
			if (f.noDeadMode_) {
				score += f.coin_ * HP.FlogCoinRate * HP.NoDeadCoinRate;
			}
			int liveTurn = f.deadTurn_ - server.state_.turn_;
			score -= (W - liveTurn) * HP.DeadTurnRate;
		}
		else {
			score += f.coin_ * HP.FlogCoinRate;

			const pint& p = gs.ToPos(f.pos_);

			if (f.noDeadMode_ || f.onNoDeadLog_) {
			}
			else {
				double prob = aliveProb_.GetHomeProb(f.pos_, state.turn_);
				VASSERT(prob <= 1);
				prob = HP.ProbGeta + (1.0 - HP.ProbGeta) * prob;

				score *= prob;
			}
		}

		const pint& p = gs.ToPos(f.pos_);

		{
			static double center = (W - 1) / 2.0;

			static double XScoreMul = 1.0 / (double)W * ScoreRate * HP.XScoreRate;
			static double FrontScoreMul = 1.0 / (double)W * ScoreRate * 0.2 * HP.FrontScoreRate;
			static double YScoreMul = 1.0 / (double)H / (double)F * ScoreRate * 0.1 * HP.YScoreRate;

			score -= fabs(p.x - center) * XScoreMul;

			if (server.state_.turn_ < homeStartTurn_) {
				int frontX = CalcFrontX(p.x, p.y);
				score += frontX * FrontScoreMul;

				score += p.y * YScoreMul;
			}
		}
		{
			int diffCount = 0;
			if (p.y - 1 >= 0) {
				if (MoveDirs[p.y] != MoveDirs[p.y - 1]) {
					++diffCount;
				}
			}
			if (p.y + 1 < H) {
				if (MoveDirs[p.y] != MoveDirs[p.y + 1]) {
					++diffCount;
				}
			}
			score += diffCount * ScoreRate * 0.1 * HP.DiffDirRate;
		}
		if (F >= 2) {
			bool sameY = false;
			REP(fi2, F) {
				if (fi2 == fi) {
					continue;
				}
				cauto& f2 = state.frogs_[fi2];
				if (f.pos_ == f2.pos_) {
					continue;
				}
				cauto& p2 = gs.ToPos(f2.pos_);
				if (p2.y == p.y) {
					sameY = true;
					break;
				}
			}
			if (sameY) {
				score -= ScoreRate * 0.1 * HP.FrogYDiffRate;
			}
		}
		if (server.state_.turn_ < outEndTurn_) {
			if (!f.IsDead()) {
				pint zeroP = gs.ToPos(f.pos_);
				zeroP.x -= MoveDirs[zeroP.y] * (state.turn_ - server.state_.turn_);
				int outY = outYMap_.Get(zeroP.y, zeroP.x);
				VASSERT(outY > -127);
				score += outY * ScoreRate * HP.OutFailRate;
			}
		}
		if (server.state_.turn_ >= homeStartTurn_) {
			pint zeroP = gs.ToPos(f.pos_);
			if (f.IsDead()) {
				zeroP.x -= MoveDirs[zeroP.y] * (f.deadTurn_ - server.state_.turn_);
			}
			else {
				zeroP.x -= MoveDirs[zeroP.y] * (state.turn_ - server.state_.turn_);
			}
			int homeY = homeYMap_.Get(zeroP.y, zeroP.x);
			VASSERT(homeY >= 0);
			double rate = (H - homeY) / (double)H;		
			VASSERT(rate <= 1);
			score *= rate;
		}

		return score;
	}

	void MakeHomeYMap2(const SimState& state, ExtGrid<s8>& homeYMap) {
		WHQue<pint> que;
		homeYMap.Init(-1);
		REP(y, H) {
			if (y == 0) {
				REP(x, W) {
					homeYMap.Set(y, x, y);
				}
			}
			else {
				if (MoveDirs[y - 1] != MoveDirs[y]) {
					FOR(x, -1, W + 1) {
						char v = state.GetSimValue(y, x);
						if (v != CT::Water) {
							homeYMap.Set(y, x, y);
							que.push(pint{ x, y });
						}
					}
				}
			}
		}

		{
			while (que.exist()) {
				pint p = que.pop();
				for (pint dir : Around4) {
					pint n = p + dir;
					if (n.y >= H) {
						continue;
					}
					if (MoveDirs[p.y] != MoveDirs[n.y]) {
						continue;
					}

					char v = state.GetSimValue(n.y, n.x);
					if (v == CT::Water) {
						continue;
					}
					if (homeYMap.Get(n.y, n.x) >= 0) {
						continue;
					}
					homeYMap.Set(n.y, n.x, homeYMap.Get(p.y, p.x) + 1);
					que.push(n);
				}
			}
		}

		que.clear();
		REP(y, H) {
			FOR(x, -1, W + 1) {
				char v = state.GetSimValue(y, x);
				if (v == CT::Water) {
					continue;
				}
				if (homeYMap.Get(y, x) < 0) {
					if (y + 1 < H) {
						if (MoveDirs[y] != MoveDirs[y + 1]) {
							homeYMap.Set(y, x, y + 2);
							que.push(pint{ x, y });
						}
					}
				}
			}
		}
		constexpr array<pint, 3> dirs = { pint{-1, 0}, pint{1,0}, pint{0,-1} };
		while (que.exist()) {
			pint p = que.pop();
			for (pint dir : dirs) {
				pint n = p + dir;
				char v = state.GetSimValue(n.y, n.x);
				if (v == CT::Water) {
					continue;
				}
				if (homeYMap.Get(n.y, n.x) >= 0) {
					continue;
				}
				homeYMap.Set(n.y, n.x, homeYMap.Get(p.y, p.x) + 1);
				que.push(n);
			}
		}

		REP(y, H) {
			FOR(x, -1, W + 1) {
				char v = state.GetSimValue(y, x);
				if (v == CT::Water) {
					continue;
				}
				if (homeYMap.Get(y, x) < 0) {
					homeYMap.Set(y, x, y + 6);
				}
			}
		}
	}

	void MakeOutYMap2(const SimState& state, ExtGrid<s8>& outYMap) {
		constexpr s8 NotChecked = -127;

		WHQue<pint> que;
		outYMap.Init(NotChecked);
		REP(y, H) {
			if (y == 0) {
				REP(x, W) {
					outYMap.Set(y, x, y);
				}
			}
			else {
				if (y + 1 >= H || MoveDirs[y + 1] != MoveDirs[y]) {
					FOR(x, -1, W + 1) {
						char v = state.GetSimValue(y, x);
						if (v != CT::Water) {
							outYMap.Set(y, x, y);
							que.push(pint{ x, y });
						}
					}
				}
			}
		}
		{
			while (que.exist()) {
				pint p = que.pop();
				for (pint dir : Around4) {
					pint n = p + dir;
					if (n.y >= H) {
						continue;
					}
					if (MoveDirs[p.y] != MoveDirs[n.y]) {
						continue;
					}

					char v = state.GetSimValue(n.y, n.x);
					if (v == CT::Water) {
						continue;
					}
					if (outYMap.Get(n.y, n.x) != NotChecked) {
						continue;
					}
					outYMap.Set(n.y, n.x, outYMap.Get(p.y, p.x) - 1);
					que.push(n);
				}
			}
		}

		que.clear();
		REP(y, H) {
			FOR(x, -1, W + 1) {
				char v = state.GetSimValue(y, x);
				if (v == CT::Water) {
					continue;
				}
				if (outYMap.Get(y, x) == NotChecked) {
					if (MoveDirs[y] != MoveDirs[y - 1]) {
						outYMap.Set(y, x, y - 2);
						que.push(pint{ x, y });
					}
				}
			}
		}
		constexpr array<pint, 3> dirs = { pint{-1, 0}, pint{1,0}, pint{0,1} };
		while (que.exist()) {
			pint p = que.pop();
			for (pint dir : dirs) {
				pint n = p + dir;
				if (n.y >= H) {
					continue;
				}
				char v = state.GetSimValue(n.y, n.x);
				if (v == CT::Water) {
					continue;
				}
				if (outYMap.Get(n.y, n.x) != NotChecked) {
					continue;
				}
				outYMap.Set(n.y, n.x, outYMap.Get(p.y, p.x) - 1);
				que.push(n);
			}
		}
		REP(y, H) {
			FOR(x, -1, W + 1) {
				char v = state.GetSimValue(y, x);
				if (v == CT::Water) {
					continue;
				}
				if (outYMap.Get(y, x) == NotChecked) {
					outYMap.Set(y, x, y - 8);
				}
			}
		}
	}

	void MakeLifeMap(const State& state, WHArr<s8>& lifeMap) {
		lifeMap.assign(WH, -1);

		HArr<s8> maxOuterLogTurn;
		maxOuterLogTurn.assign(H, -1);
		FOR(y, 1, H) {
			int dir = MoveDirs[y];
			int x = (dir < 0) ? (W - 1) : 0;
			int visibleWaterCount = -1;
			REP(i, W) {
				int pos = gs.ToId(x, y);
				if (state.grid_[pos] != CT::Water) {
					visibleWaterCount = i;
					break;
				}
				x += dir;
			}
			VASSERT(visibleWaterCount >= 0);
			maxOuterLogTurn[y] = KW - visibleWaterCount + 1;
		}

		auto IsMaxTurnLog = [&](int x, int y) {
			int front = x + MoveDirs[y];
			if (front >= 0 && front < W && state.grid_[gs.ToId(front, y)] != CT::Water) {
				int frontX = CalcFrontX(x, y);
				if (y - 1 >= 1 && MoveDirs[y] != MoveDirs[y - 1]) {
					if (frontX >= maxOuterLogTurn[y - 1]) {
						return true;
					}
				}
				if (y + 1 < H && MoveDirs[y] != MoveDirs[y + 1]) {
					if (frontX >= maxOuterLogTurn[y + 1]) {
						return true;
					}
				}
			}
			return false;
		};

		REP(y, H) {
			if (y == 0) {
				REP(x, W) {
					if (state.grid_[gs.ToId(x, y)] != CT::Water) {
						lifeMap[gs.ToId(x, y)] = -1;
					}
				}
			}
			else {
				bool maxLog = false;
				RREP(frontX, W) {
					int x = FromFrontX(frontX, y);
					if (state.grid_[gs.ToId(x, y)] != CT::Water) {
						if (maxLog) {
							lifeMap[gs.ToId(x, y)] = 1;
						}
						else if (IsMaxTurnLog(x, y)) {
							lifeMap[gs.ToId(x, y)] = 2;
							maxLog = true;
						}
						else {
							lifeMap[gs.ToId(x, y)] = -1;
							maxLog = false;
						}
					}
					else {
						maxLog = false;
					}
				}
			}
		}

	}

	u64 GetCellHash(int t, int y, int x, char v) {
		VASSERT(x >= 0 && x < W);
		static int EW = W + 2;
		int zeroX = x - MoveDirs[y] * t;
		VASSERT(zeroX >= -1 && zeroX <= W);
		int zpos = y * EW + (zeroX + 1);
		int zv = 0;
		if (v == CT::Frog) {
			zv = 1;
		}
		else if (v == CT::Coin) {
			zv = 2;
		}
		return zob_.Hash(zpos, zv);
	}
	u64 GetCellHash(int t, int pos, char v) {
		pint p = gs.ToPos(pos);
		return GetCellHash(t, p.y, p.x, v);
	}
};


struct Main {
    void Run(int argc, const char* argv[]) {
        ChronoTimer timer;
        server.InitInput(timer);

        static Solver4 solver;
        timer.StartMs(TIME_LIMIT);

        solver.Run(timer);

        server.Finalize();
    }
};
