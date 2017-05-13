namespace utils {

inline int kroneckerDelta(const uint i, const uint j) {
        if (i == j) {
            return 1;
        } else {
            return 0;
        }
}

} // namespace utils
