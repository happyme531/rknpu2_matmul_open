#ifndef RK_NPU_I4_LAYOUT_H
#define RK_NPU_I4_LAYOUT_H
#include <algorithm>
namespace rknpu2_matmul_open::detail {
enum class I4InputLayout { Native, Panel8 };
// Panel8 tiles contain whole groups of eight physical rows. A final group
// smaller than eight uses native layout, with no padded logical tokens.
inline int i4_m_rows(int remaining, int tile, I4InputLayout layout) {
    const int rows = std::min(remaining, tile);
    return layout == I4InputLayout::Panel8 && rows >= 8 ? rows / 8 * 8 : rows;
}
inline int i4_panel_entries(int k) {
    const int tail = k % 128;
    return 8 * (k / 128) + (tail == 0 ? 0 : tail <= 32 ? 2 : tail <= 64 ? 4 : 8);
}
inline int i4_panel_banks(int rows, int k) {
    return (rows / 8 * i4_panel_entries(k) * 64 + 32767) / 32768;
}
} // namespace rknpu2_matmul_open::detail
#endif
