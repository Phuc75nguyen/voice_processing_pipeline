import java.util.*;

public class ThreeSum {
    public static List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums); // B1: Sắp xếp mảng tăng dần
        List<List<Integer>> result = new ArrayList<>();
        int n = nums.length;

        // B2: Duyệt từng phần tử làm mốc
        for (int i = 0; i < n; i++) {
            // Bỏ qua phần tử trùng lặp
            if (i > 0 && nums[i] == nums[i - 1]) continue;

            int left = i + 1;
            int right = n - 1;

            // B3: Dùng hai con trỏ để tìm cặp còn lại
            while (left < right) {
                int total = nums[i] + nums[left] + nums[right];

                if (total == 0) {
                    // Thêm bộ 3 vào kết quả
                    result.add(Arrays.asList(nums[i], nums[left], nums[right]));
                    left++;
                    right--;

                    // Bỏ qua các phần tử trùng ở bên trái và phải
                    while (left < right && nums[left] == nums[left - 1]) left++;
                    while (left < right && nums[right] == nums[right + 1]) right--;
                } else if (total < 0) {
                    left++;
                } else {
                    right--;
                }
            }
        }

        return result;
    }

    // --- Hàm main để test ---
    public static void main(String[] args) {
        int[] example1 = {-1, 0, 1, 2, -1, 4};
        int[] example2 = {0, 1, 1};
        int[] example3 = {0, 0, 0};

        System.out.println("Ví dụ 1: " + threeSum(example1)); // [[-1, -1, 2], [-1, 0, 1]]
        System.out.println("Ví dụ 2: " + threeSum(example2)); // []
        System.out.println("Ví dụ 3: " + threeSum(example3)); // [[0, 0, 0]]
    }
}
