# Bài toán: Tìm tất cả bộ 3 số trong mảng có tổng bằng 0, không trùng nhau

from typing import List

def three_sum(nums: List[int]) -> List[List[int]]:
    # B1: Sắp xếp mảng để dễ xử lý trùng lặp và dùng hai con trỏ
    nums.sort()
    result = []
    n = len(nums)

    # B2: Duyệt từng phần tử làm mốc (nums[i])
    for i in range(n):
        # Nếu phần tử hiện tại giống phần tử trước → bỏ qua để tránh trùng
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        # Hai con trỏ left và right
        left, right = i + 1, n - 1

        # B3: Dịch chuyển hai con trỏ tìm cặp có tổng = -nums[i]
        while left < right:
            total = nums[i] + nums[left] + nums[right]

            if total == 0:
                # Ghi nhận bộ 3 thỏa điều kiện
                result.append([nums[i], nums[left], nums[right]])

                # Dịch con trỏ sang tránh trùng lặp
                left += 1
                right -= 1
                while left < right and nums[left] == nums[left - 1]:
                    left += 1
                while left < right and nums[right] == nums[right + 1]:
                    right -= 1

            elif total < 0:
                left += 1  # Tổng nhỏ hơn 0 → tăng left để tổng lớn hơn
            else:
                right -= 1  # Tổng lớn hơn 0 → giảm right để tổng nhỏ hơn

    return result


# --- Kiểm thử ---
print("Ví dụ 1:", three_sum([-1, 0, 1, 2, -1, 4]))  # [[-1, -1, 2], [-1, 0, 1]]
print("Ví dụ 2:", three_sum([0, 1, 1]))             # []
print("Ví dụ 3:", three_sum([0, 0, 0]))             # [[0, 0, 0]]
