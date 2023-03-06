import javafx.util.Pair;
import java.util.*;

import static java.util.List.*;

public class Main {
    public static void main(String[] args) {

    }

    //89. Gray Code - Medium - Bit Shifts
    public static List<Integer> grayCode(int n) {
        List<Integer> res = new ArrayList<>();
        res.add(0);
        for(int i = 1; i <= n; i++)
        {
            int size = res.size() - 1;
            int mask = 1 << (i - 1);
            for(int j = size; j >= 0; j--)
            {
                res.add( mask + res.get(j));
            }
        }
        return res;
    }
    //371. SUM OF TWO INTEGERS
    public static int getSum(int a, int b) {
        while(b != 0)
        {
            int answer = a ^ b;
            int carry = (a & b) << 1;
            a = answer;
            b = carry;
        }
        return a;
    }
    // 190. RESERVE BITS - EASY - MASK AND BIT SHIFT
    public static int reverseBits(int num) {
        num = ((num & 0xffff0000) >>> 16) | ((num & 0x0000ffff) << 16);
        num = ((num & 0xff00ff00) >>> 8) | ((num & 0x00ff00ff) << 8);
        num = ((num & 0xf0f0f0f0) >>> 4) | ((num & 0x0f0f0f0f) << 4);
        num = ((num & 0xcccccccc) >>> 2) | ((num & 0x33333333) << 2);
        num = ((num & 0xaaaaaaaa) >>> 1) | ((num & 0x55555555) << 1);
        // n = ((n & 0b11111111111111110000000000000000) >>> 16) | ((n & 0b00000000000000001111111111111111) << 16);
        // n = ((n & 0b11111111000000001111111100000000) >>> 8)  | ((n & 0b00000000111111110000000011111111) << 8);
        // n = ((n & 0b11110000111100001111000011110000) >>> 4)  | ((n & 0b00001111000011110000111100001111) << 4);
        // n = ((n & 0b11001100110011001100110011001100) >>> 2)  | ((n & 0b00110011001100110011001100110011) << 2);
        // n = ((n & 0b10101010101010101010101010101010) >>> 1)  | ((n & 0b01010101010101010101010101010101) << 1);
        return num;
    }
    //975. ODD EVEN JUMP - Hard
    public static int oddEvenJumps(int[] arr) {
        int res = 1; //alway reach the end staring at last index
        int n = arr.length;
        boolean[][] dp = new boolean[arr.length][2];
        TreeMap<Integer, Integer> map = new TreeMap<>();
        map.put(arr[n - 1], n - 1);
        dp[n - 1][0] = true; //odd step
        dp[n - 1][1] = true; //even step

        for(int i = n - 2; i >=0; i--)
        {
            //take odd step
            Integer nextGreater = map.ceilingKey(arr[i]);
            if(nextGreater != null)
            {
                dp[i][0] = dp[map.get(nextGreater)][1];
            }

            //take even step
            Integer nextLower = map.floorKey(arr[i]);
            if(nextLower != null)
            {
                dp[i][1] = dp[map.get(nextLower)][0];
            }
            map.put(arr[i], i);

            if(dp[i][0])
                res++;
        }
        return res;
    }
    //2454. NEXT GREATER ELEMENT IV
    public static int[] secondGreaterElement(int[] nums) {
        Deque<Integer> stack1 = new ArrayDeque<>();
        Deque<Integer> stack2 = new ArrayDeque<>();
        Deque<Integer> temp = new ArrayDeque<>();
        int[] res = new int[nums.length];
        Arrays.fill(res, -1);

        for(int i = 0; i < nums.length; i++)
        {
            while(!stack2.isEmpty() && nums[stack2.peek()] < nums[i])
            {
                res[stack2.pop()] = nums[i];
            }

            while(!stack1.isEmpty() && nums[stack1.peek()] < nums[i])
            {
                temp.push(stack1.pop());
            }

            while(!temp.isEmpty())
                stack2.push(temp.pop());

            stack1.push(i);
        }
        return res;
    }

    //556. NEXT GREATER ELEMENT III - MEDIUM
    public static int nextGreaterElement(int n) {
        char[] num = String.valueOf(n).toCharArray();
        int i = num.length - 2;
        int j = num.length - 1;

        while(i >= 0 && num[i] >= num[i + 1])
            i--;

        if(i < 0)
            return -1;

        while(j >= 0 && num[j] <= num[i])
            j--;

        nextGreaterElementSwap(num, i, j);
        nextGreaterElementReverse(num, i + 1);

        try{
            return Integer.parseInt(new String(num));
        }catch (Exception e)
        {
            return - 1;
        }
    }
    public static void nextGreaterElementSwap(char[] num, int i, int j)
    {
        char temp = num[i];
        num[i] = num[j];
        num[j] = temp;
    }
    public static void nextGreaterElementReverse(char[] num, int i)
    {
        int j = num.length - 1;
        while(i < j)
        {
            nextGreaterElementSwap(num, i, j);
            i++;
            j--;
        }
    }
    //310. MINIMUM HEIGHT TREES - MEDIUM - TOPOLOGICAL SORTING WITH KAHN'S ALGO - CUT LEAVES
    public static List<Integer> findMinHeightTrees(int n, int[][] edges) {
        int[] indegrees = new int[n];
        Map<Integer, List<Integer>> adjacencies = new HashMap<>();
        List<Integer> leaves = new ArrayList<>(); //result
        //edge case
        if(n < 2)
        {
            for(int i = 0; i < n; i++)
            {
                leaves.add(i);
            }
            return leaves;
        }

        for(int[] edge: edges)
        {
            adjacencies.putIfAbsent(edge[0], new ArrayList<>());
            adjacencies.get(edge[0]).add(edge[1]);
            adjacencies.putIfAbsent(edge[1], new ArrayList<>());
            adjacencies.get(edge[1]).add(edge[0]);
            indegrees[edge[0]]++;
            indegrees[edge[1]]++;
        }

        for(int i = 0; i < n; i++)
        {
            if(indegrees[i] == 1)
            {
                leaves.add(i);
            }
        }
        int remainingNodes = n;
        while(remainingNodes > 2)
        {
            remainingNodes -= leaves.size();
            List<Integer> newLeaves = new ArrayList<>();
            for(int leave: leaves)
            {
                for(int node: adjacencies.get(leave))
                {
                    indegrees[node]--;
                    if(indegrees[node] == 1)
                    {
                        newLeaves.add(node);
                    }
                }
            }
            leaves = newLeaves;
        }
        return leaves;
    }
    //269. ALIEN DICTIONARY - HARD - KAHN'S ALGORITHM - TOPOLOGICAL SORTING
    public static String alienOrder(String[] words) {
        if(words == null || words.length == 0) return "";
        //1. Init inDegree & topoMap
        HashMap<Character, Integer> inDegree = new HashMap<>();
        HashMap<Character, List<Character>> topoMap = new HashMap<>();
        for(String word : words)
            for(char c : word.toCharArray()) {
                inDegree.put(c, 0);
                topoMap.put(c, new ArrayList<Character>());
            }
        //2. Build Map
        for(int i = 0; i < words.length - 1; i++) {
            String w1 = words[i], w2 = words[i + 1];
            //check if w2 is a prefix of w1
            if(w1.length() > w2.length() && w1.startsWith(w2))
                return "";
            for(int j = 0; j < Math.min(w1.length(), w2.length()); j++) {
                char parent = w1.charAt(j), child = w2.charAt(j);
                if(parent != child) {
                    inDegree.put(child, inDegree.get(child) + 1);
                    topoMap.get(parent).add(child);
                    break;
                }
            }
        }
        //3. Topo sort
        StringBuilder res = new StringBuilder();
        while(!inDegree.isEmpty()) {
            boolean flag = false;
            for(Character c : inDegree.keySet()) {
                if(inDegree.get(c) == 0) {
                    flag = true;
                    res.append(c);
                    List<Character> children = topoMap.get(c);
                    for(Character ch : children)
                        inDegree.put(ch, inDegree.get(ch) - 1);
                    inDegree.remove(c);
                    break;
                }
            }
            if(flag == false)
                return "";
        }
        return res.toString();
    }

    //210. COURSE SCHEDULE II - MEDIUM - Topological Shorting - Kahn's Algorithm
    public static int[] findOrder(int numCourses, int[][] prerequisites) {
        int[] res = new int[numCourses];
        int[] inDegree = new int[numCourses];
        int index = 0; //index of the result
        List<List<Integer>> adjacencies = new ArrayList<List<Integer>>();
        Deque<Integer> queue = new ArrayDeque<>(); //contains all inDegree = 0;
        for(int i = 0; i < numCourses; i++)
        {
            adjacencies.add(new ArrayList<>());;
        }
        for(int[] pre: prerequisites)
        {
            inDegree[pre[0]]++;
            adjacencies.get(pre[1]).add(pre[0]);
        }
        for(int i = 0; i < numCourses; i++)
        {
            if(inDegree[i] == 0)
            {
                queue.offer(i);
            }
        }

        while(!queue.isEmpty())
        {
            int curr = queue.poll();
            res[index++] = curr;
            for(int course: adjacencies.get(curr))
            {
                inDegree[course]--;
                if(inDegree[course] == 0)
                {
                    queue.offer(course);
                }
            }

            if(queue.isEmpty() && index != numCourses)
                return new int[]{};
        }
        return (queue.isEmpty() && index != numCourses)? new int[]{}: res;
    }
    //1631. PATH WITH MINIMUM EFFORT - MEDIUM - DIJKSTRA ALGORITHM
    public static int minimumEffortPath(int[][] heights) {
        int row = heights.length;
        int col = heights[0].length;
        int[][] costs = new int[row][col];
        boolean[][] visited = new boolean[row][col];
        int[][] direction = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        for(int i = 0; i < heights.length; i++)
        {
            Arrays.fill(costs[i], Integer.MAX_VALUE);
        }
        costs[0][0] = 0;
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) ->a[2] - b[2]);
        pq.offer(new int[]{0, 0, 0});

        while(pq.size() > 0)
        {
            int[] curr = pq.poll();
            int x = curr[0];
            int y = curr[1];
            int dist = curr[2];
            visited[x][y] = true;
            for(int[] dir: direction)
            {
                int xPos = x + dir[0];
                int yPos = y + dir[1];

                if((xPos < 0 || xPos >= row || yPos < 0 || yPos >= col)  || visited[xPos][yPos])
                    continue;

                int diff= Math.abs(heights[x][y] - heights[xPos][yPos]);
                int max = Math.max(diff, costs[x][y]);

                if(max < costs[xPos][yPos])
                {
                    costs[xPos][yPos] = max;
                    pq.offer(new int[]{xPos, yPos, max});
                }
            }
        }
        return costs[row - 1][col - 1];
    }
    //787. CHEAPEST FLIGHT WITHIN K STOPS - MEDIUM - BELLMAN FORD ALGORITHM
    public static int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {

        //distance from source to all other nodes
        int[] dist = new int[n];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[src] = 0;
        //run only k times since we want shortest distance in k stops
        for(int i = 0; i <= k; i++)
        {
            int[] temp = Arrays.copyOf(dist, n);

            for(int[] flight : flights)
            {
                if(dist[flight[0]] != Integer.MAX_VALUE)
                {
                    temp[flight[1]] = Math.min(temp[flight[1]], dist[flight[0]] + flight[2]);
                }
            }
            dist = temp;
        }
        return dist[dst] == Integer.MAX_VALUE? -1 : dist[dst];
    }
    //90. subset II - contains duplicates- Medium - Back Track
    public static List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> sub = new ArrayList<>();
        Arrays.sort(nums);

        res.add(sub);
        subsetsWithDupHelper(nums, 0, sub, res);
        return res;
    }

    public static void subsetsWithDupHelper(int[] nums, int start, List<Integer> sub, List<List<Integer>> res )
    {
        for(int i = start; i < nums.length; i++)
        {
            if(i != start && nums[i - 1] == nums[i])
                continue;

            sub.add(nums[i]);
            res.add(new ArrayList<>(sub));
            subsetsWithDupHelper(nums, i + 1, sub, res);
            sub.remove(sub.size() - 1);
        }
    }
    //71. Simplify Path - Medium
    public static String simplifyPath(String path) {
        String[] names = path.split("/");
        Deque<String> stack = new ArrayDeque<>();
        for(String name: names)
        {

            if(name.equals("."))
                continue;

            if(name.equals(".."))
            {
                if(!stack.isEmpty())
                {
                    stack.pop();
                }
            }else
            {
                stack.push(name);
            }
        }

        StringBuilder res = new StringBuilder();
        for(String s: stack)
        {
            res.append("/");
            res.append(s);
        }


        return res.toString();
    }
    //61. Rotate List - Medium
    public static ListNode rotateRight(ListNode head, int k) {
        if(head == null || head.next == null || k == 0)
            return head;

        ListNode next = head;
        int i = 0; //the length of the linked list
        for(i = 1; next.next != null; i++)
        {
            next = next.next;
        }
        next.next = head; //connect tail to head; now this is a circle linked list

        k = k % i - 1; //new head position
        ListNode start = head;
        for(int j = 0; j < k; j++)
        {
            start = start.next;
        }

        head = start.next; // new head;
        start.next = null; //new tail;

        return head;
    }
    //48. Rotate Image - Medium
    public static void rotate(int[][] matrix) {
        rotateTranspose(matrix);
        rotateReverse(matrix);
    }

    public static void rotateTranspose(int[][] matrix)
    {
        for(int i = 0; i < matrix.length; i++)
        {
            for(int j = i + 1; j < matrix[i].length; j++)
            {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
    }

    public static void rotateReverse(int[][] matrix)
    {
        for(int i = 0; i < matrix.length; i++)
        {
            int first = 0;
            int last = matrix.length - 1;
            while(first < last)
            {
                int temp = matrix[i][first];
                matrix[i][first] = matrix[i][last];
                matrix[i][last] = temp;
                first++;
                last--;
            }
        }
    }
    //Permutation II (input array contains duplicates)- Medium - Backtrack with group of numbers
    public static List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Map<Integer, Integer> counter = new HashMap<Integer, Integer>();
        for(int num : nums)
        {
            Integer value = counter.getOrDefault(num, 0);
            counter.put(num, value + 1);
        }
        permuteUniqueHelper(nums, new ArrayList<>(), res, counter);

        return res;
    }

    public static void permuteUniqueHelper(int[] nums, List<Integer> permute, List<List<Integer>> res, Map<Integer, Integer> counter)
    {
        if(permute.size() == nums.length)
        {
            res.add(new ArrayList<Integer>(permute));
        }

        for(Map.Entry<Integer,Integer> entry: counter.entrySet())
        {
            Integer key = entry.getKey();
            Integer value = entry.getValue();
            if(value == 0)
                continue;

            permute.add(key);
            counter.put(key, value - 1);
            permuteUniqueHelper(nums, permute, res, counter);
            permute.remove(permute.size() - 1);
            counter.put(key, counter.get(key) + 1);
        }
    }
    //46. Permutations - Medium
    public static List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        permuteHelper(nums, new ArrayList<Integer>(), res );
        return res;
    }

    public static void permuteHelper(int[] nums, List<Integer> permutation, List<List<Integer>> res)
    {
        if(permutation.size() == nums.length)
        {
            res.add(new ArrayList<Integer>(permutation));
            return;
        }

        for(int i = 0; i < nums.length; i++)
        {
            if(!permutation.contains(nums[i]))
            {
                permutation.add(nums[i]);
                permuteHelper(nums, permutation, res);
                permutation.remove(permutation.size() - 1);
            }
        }
    }
    //43. Multiply Strings- Medium
    public static String multiply(String num1, String num2) {
        int m = num1.length();
        int n = num2.length();
        int[] pos = new int[m + n];
        for(int i = m -1; i >= 0; i--)
        {
            for(int j = n - 1; j >= 0; j--)
            {
                int mul = (num1.charAt(i) - '0') * (num2.charAt(j) - '0');
                int pos1 = i + j;
                int pos2 = i + j + 1;

                int sum = mul + pos[pos2];
                pos[pos2] = sum % 10;
                pos[pos1] += sum / 10;
            }
        }

        StringBuilder res = new StringBuilder();
        for(int i = 0; i < pos.length; i++)
        {
            if(pos[i] == 0 && res.length() == 0)
            {
                continue;
            }
            res.append(pos[i]);
        }
        return res.length() == 0? "0": res.toString();
    }
    //40. Combination Sum II - Medium
    public static List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(candidates);
        combinationSum2Helper(candidates, target, 0, new ArrayList<Integer>(), res);
        return res;
    }

    public static void combinationSum2Helper(int[] candidates, int target, int start, List<Integer> combination, List<List<Integer>> res)
    {
        if(target == 0)
        {
            res.add(new ArrayList<>(combination));
            return;
        }else if(target < 0){
            return;
        }

        for(int i = start; i < candidates.length; i++)
        {
            //avoid duplicates results
            if(i > start && candidates[i] == candidates[i - 1])
                continue;

            combination.add(candidates[i]);
            combinationSum2Helper(candidates, target - candidates[i], i + 1, combination, res);
            combination.remove(combination.size() - 1);
        }
    }
    //39. Combination Sum - Medium
    public static List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        combinationSumHelper(candidates, target, 0, new ArrayList<>(), res);
        return res;
    }

    public static void combinationSumHelper(int[] candidates, int target, int start, List<Integer> combination, List<List<Integer>> res)
    {
        if(target == 0)
        {
            res.add(new ArrayList<>(combination));
            return;
        }else if(target < 0)
        {
            return;
        }

        for(int i = start; i < candidates.length; i++)
        {
            combination.add(candidates[i]);
            combinationSumHelper(candidates, target - candidates[i], i, combination, res);
            combination.remove(combination.size() - 1);
        }
    }
    //38. Count and Say - Medium
    public static String countAndSay(int n) {
        String res = "1";

        for(int i = 1; i < n; i++)
        {
            res = countAndSayHelper(res);
        }

        return res;
    }

    public static String countAndSayHelper(String res)
    {
        StringBuilder sb = new StringBuilder();
        int count = 1;
        char say = res.charAt(0);
        for(int i = 1; i < res.length(); i++)
        {
            if(say == res.charAt(i))
            {
                count++;
            }
            else
            {
                sb.append(count).append(say);
                say = res.charAt(i);
                count = 1;
            }
        }
        sb.append(count).append(say);
        return sb.toString();
    }
    //34. Find the First and Last Position of Element in a sorted Array - Medium
    public static int[] searchRange(int[] nums, int target) {
        if(nums.length == 0)
            return new int[]{-1, -1};

        int leftMost = searchRangeHelper(nums, target, true);
        if(leftMost == -1)
            return new int[]{-1, -1};

        int rightMost = searchRangeHelper(nums, target, false);
        return new int[]{leftMost, rightMost};
    }

    public static int searchRangeHelper(int[] nums, int target, boolean isFirst)
    {
        int left = 0;
        int right = nums.length - 1;

        while(left <= right)
        {
            int mid = left - (left - right)/2;
            if(nums[mid] == target)
            {
                if(isFirst)
                {
                    if((mid == left) || (nums[mid] != nums[mid - 1]))
                        return mid;
                    right = mid - 1;
                }
                else
                {
                    if((mid == right) || (nums[mid] != nums[mid + 1]))
                        return mid;
                    left = mid + 1;
                }
            }
            else if(nums[mid] > target)
            {
                right = mid -1 ;
            }
            else
                left = mid + 1;
        }
        return -1;
    }
    //16. 3sum closest - medium
    public static int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int closest = Integer.MAX_VALUE;
        int res = Integer.MAX_VALUE;
        int size = nums.length - 1;

        for(int j = 0; j <= size - 1; j++)
        {
            int left = j + 1;
            int right = size;
            while(left < right)
            {
                int sum = nums[j] + nums[left] + nums[right];
                if(Math.abs(closest) > Math.abs(target - sum))
                {
                    closest = target - sum;
                    res = sum;
                }
                else if(sum < target)
                    left++;
                else
                    right--;
            }
        }
        return res;
    }
    //18. 4Sum - Medium
    public static List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        return kSum(nums, target, 0, 4);
    }
    public static List<List<Integer>> kSum(int[] nums, long target, int start, int k) {
        List<List<Integer>> res = new ArrayList<>();
        // If we have run out of numbers to add, return res.
        if (start == nums.length) {
            return res;
        }
        if (k == 2) {
            return twoSum(nums, target, start);
        }
        for (int i = start; i < nums.length; ++i) {
            if (i == start || nums[i - 1] != nums[i])
            {
                List<List<Integer>> temp = kSum(nums, target - nums[i], i + 1, k - 1);
                for (List<Integer> subset : temp) {
                    res.add(new ArrayList<>(Arrays.asList(nums[i])));
                    res.get(res.size() - 1).addAll(subset);
                }
            }
        }
        return res;
    }

    public static List<List<Integer>> twoSum(int[] nums, long target, int start) {
        List<List<Integer>> res = new ArrayList<>();
        int lo = start, hi = nums.length - 1;

        while (lo < hi) {
            int currSum = nums[lo] + nums[hi];
            if (currSum < target || (lo > start && nums[lo] == nums[lo - 1])) {
                ++lo;
            } else if (currSum > target || (hi < nums.length - 1 && nums[hi] == nums[hi + 1])) {
                --hi;
            } else {
                res.add(Arrays.asList(nums[lo++], nums[hi--]));
            }
        }
        return res;
    }
    //15. 3Sum And Two sum - Medium
    public static List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for(int i = 0; i < nums.length; i++)
        {
            if(i == 0 || nums[i] != nums[i - 1])
            {
                twoSum(nums, i, res);
            }
        }
        return res;
    }

    public static void twoSum(int[] numbers, int target, List<List<Integer>> res) {
        int i = target + 1;
        int j = numbers.length - 1;
        while(i < j)
        {
            int sum = numbers[i] + numbers[j] + numbers[target];
            if(sum == 0)
            {
                res.add(Arrays.asList(numbers[target], numbers[i++], numbers[j--]));
                while(i < j && numbers[i] == numbers[i - 1])
                    i++;
            }
            else if(sum > 0)
            {
                j--;
            }else
            {
                i++;
            }
        }
    }
    //167. Two Sum II - Input Array is Sorted - Mediums
    public static int[] twoSum(int[] numbers, int target) {
        int i = 0;
        int j = numbers.length - 1;
        while(i < j)
        {
            int sum = numbers[i] + numbers[j];
            if(sum == target)
            {
                return new int[]{i + 1, j + 1};
            }
            else if(sum > target)
            {
                j--;
            }else
            {
                i++;
            }
        }
        return new int[]{-1, -1};
    }
    //7. Reverse Integer - Medium
    public static int reverse(int x) {
        int rev = 0;
        while(x!= 0)
        {
            int rem = x % 10;
            x /= 10;
            if(rev > Integer.MAX_VALUE/10 || (rev == Integer.MAX_VALUE/10 && rem > 7) ||
                    rev < Integer.MIN_VALUE/10 || (rev == Integer.MIN_VALUE/10 && rem < -8))
                return 0;

            rev = rev * 10 + rem;
        }
        return rev;
    }
    //6. Zizzag conversion - Medium
    public static String convert(String s, int numRows) {
        if(numRows == 1)
            return s;

        int maxRow = Math.min(s.length(), numRows);
        StringBuilder[] rows = new StringBuilder[maxRow];
        for(int i = 0; i < maxRow; i++)
        {
            rows[i] = new StringBuilder();
        }
        int currentRow = 0;
        boolean goingDown = false;

        for(char c : s.toCharArray())
        {
            rows[currentRow].append(c);
            if(currentRow == 0 || currentRow == maxRow - 1)
                goingDown = !goingDown;
            currentRow += goingDown? 1 : -1;
        }
        StringBuilder res = new StringBuilder();
        for(StringBuilder sb : rows)
        {
            res.append(sb);
        }
        return res.toString();
    }
    //31. Next Permutation - Medium
    public  static void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        while(i >= 0 && nums[i] >= nums[i + 1])
        {
            i--;
        }

        if(i >= 0)
        {
            int j = nums.length - 1;
            while(j >= 0 && nums[j] <= nums[i])
                j--;

            nextPermutationSwap(nums, i, j);
        }
        nextPermutationReverse(nums, i + 1, nums.length - 1);
    }

    public  static void nextPermutationSwap(int[] nums, int i, int j)
    {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public  static void nextPermutationReverse(int[] nums, int i, int j)
    {
        while(i < j)
        {
            nextPermutationSwap(nums, i++,j--);
        }
    }
    //300. Longest increase subsequence - Medium - Bottom Up DP
    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);

        for(int i = nums.length - 1; i >= 0; i--)
        {
            for(int j = i + 1; j< nums.length; j++)
            {
                if(nums[i] < nums[j])
                    dp[i] = Math.max(dp[i], dp[j] + 1);
            }
        }

        int max = 0;
        for(int num : dp)
            max = Math.max(num,max);

        return max;
    }

    //139. Word Break - Medium - Top down Dp without Set
    public static boolean wordBreak(String s, List<String> wordDict) {
        Boolean[] memo = new Boolean[s.length()];

        return wordBreakHelper(s, 0, wordDict, memo);
    }

    public static boolean wordBreakHelper(String s, int index, List<String> wordDict,Boolean[] memo)
    {
        if(index == s.length())
        {
            return true;
        }

        if(memo[index] != null)
            return memo[index];

        for(int i = 0; i < wordDict.size(); i++)
        {
            String word = wordDict.get(i);
            if(index + word.length() <= s.length())
            {
                String partialWord = s.substring(index, index + word.length());

                if(word.equals(partialWord) && wordBreakHelper(s, index + word.length(), wordDict, memo))
                {
                    return memo[index] = true;
                }
            }
        }

        return memo[index] = false;
    }

    //1335. Minimum Difficult of a job Schedule - Hard - Top Down DP
    public static int minDifficulty(int[] jobDifficulty, int d) {
        if(jobDifficulty.length < d)
            return -1;

        Integer[][] memo = new Integer[jobDifficulty.length][d + 1];
        return minDifficultyHelper(jobDifficulty, 0, d - 1, memo);
    }

    public static int minDifficultyHelper(int[] jobDifficulty, int index, int d, Integer[][] memo)
    {
        //this is the last day, we have to finish all the job
        if(d == 0)
        {
            int max = jobDifficulty[index];
            for(int i = index + 1; i < jobDifficulty.length; i++)
            {
                max = Math.max(max, jobDifficulty[i]);
            }

            return max;
        }

        if(memo[index][d] != null)
            return memo[index][d];

        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;

        for(int i = index; i < jobDifficulty.length - d; i++)
        {
            max = Math.max(jobDifficulty[i], max);
            min = Math.min(min, max + minDifficultyHelper(jobDifficulty, i + 1, d - 1, memo));
        }

        return memo[index][d] = min;
    }
    //221. Maximal Square - Medium - Bottom Up DB
    public static int maximalSquare(char[][] matrix) {
        int[][] memo = new int[matrix.length + 1][matrix[0].length + 1];
        int length = 0;

        for(int i = 1; i <= matrix.length; i++)
        {
            for(int j = 1; j <= matrix[0].length; j++)
            {
                if(matrix[i- 1][j- 1] == '1')
                {
                    int top = memo[i - 1][j];
                    int left = memo[i][j - 1];
                    int topLeft = memo[i - 1][j - 1];

                    memo[i][j] = Math.min(Math.min(top, left), topLeft) + 1;
                    length = Math.max(length, memo[i][j]);
                }
            }
        }

        return length * length;
    }
    //1143. Longest Common Subsequence - Medium - Top Down DP
    public int longestCommonSubsequence_BottomUpDP(String text1, String text2) {
        int[][] memo = new int[text1.length() + 1][text2.length() + 1];

        for(int i = text1.length() - 1; i >= 0; i--)
        {
            for(int j = text2.length() - 1; j >= 0; j--)
            {
                if(text1.charAt(i) == text2.charAt(j))
                {
                    memo[i][j] = 1 + memo[i + 1][j + 1];
                }
                else
                {
                    memo[i][j] = Math.max(memo[i][j + 1], memo[i + 1][j]);
                }
            }
        }

        return memo[0][0];
    }
    //1143. Longest Common Subsequence - Medium - Top Down DP
    public int longestCommonSubsequence(String text1, String text2) {
        return longestCommonSubsequenceHelper(text1, text2, 0, 0, new Integer[text1.length()][text2.length()]);
    }

    public int longestCommonSubsequenceHelper(String text1, String text2, int i, int j, Integer[][] memo)
    {
        if(i == text1.length() || j == text2.length())
            return 0;

        if(memo[i][j] != null)
            return memo[i][j];

        if(text1.charAt(i) == text2.charAt(j))
        {
            return 1 + longestCommonSubsequenceHelper(text1, text2, i + 1, j + 1, memo);
        }
        else
        {
            int first = longestCommonSubsequenceHelper(text1, text2, i, j + 1, memo);
            int second = longestCommonSubsequenceHelper(text1, text2, i + 1, j, memo);

            return memo[i][j] = Math.max(first, second);
        }
    }
    //1770. Maximum Score from Performing Multiplication Operations - Medium -Using DP
    public int maximumScore(int[] nums, int[] multipliers) {
        int m = nums.length;
        int n = multipliers.length;
        int left = 0;
        int i = 0;
        //right = m - 1 - (i - left);
        int[][] dp = new int[multipliers.length][nums.length];

        maximumScoreHelper(nums, multipliers, dp, m, n, i, left);

        return dp[0][0];
    }

    public int maximumScoreHelper(int[] nums, int[] multipliers, int[][] dp, int m, int n, int i, int left)
    {
        if(i == n)
            return 0;

        if(dp[i][left] == 0)
        {
            int chooseLeft = multipliers[i]  * nums[left] + maximumScoreHelper(nums, multipliers, dp, m, n, i + 1, left + 1);
            int chooseRight = multipliers[i] * nums[m - 1 - (i - left)] + maximumScoreHelper(nums, multipliers, dp, m, n, i + 1, left);
            dp[i][left] = Math.max(chooseLeft, chooseRight);
        }

        return dp[i][left];
    }
    //740. Delete and Earn - Medium - Using DP
    //Convert it to a problem similar to house robber.
    public int deleteAndEarn(int[] nums) {
        int[] count = new int[10001];
        for(int i = 0; i < nums.length; i++)
        {
            count[nums[i]] += nums[i];
        }

        int[] total = new int[10001];
        total[0] = count[0];
        total[1] = Math.max(count[0], count[1]);

        for(int i = 2; i < 10001; i++)
        {
            total[i] = Math.max(count[i] + total[i - 2], total[i - 1]);
        }

        return total[10000];
    }
    //746. Min Cost Climbing Stairs - Easy - Using DP
    public int minCostClimbingStairs(int[] cost) {
        int pay[] = new int[cost.length + 1];

        for(int i = 2; i <= cost.length; i++)
        {
            int oneStep = pay[i - 1] + cost[i - 1];
            int twoStep = pay[i - 2] + cost[i - 2];

            pay[i] = Math.min(oneStep, twoStep);
        }

        return pay[cost.length];
    }
    //198. House Robber - Medium - Using Dp
    public int rob(int[] nums) {
        if(nums.length == 1)
            return nums[0];

        int[] money = new int[nums.length];

        money[0] = nums[0];
        money[1] = Math.max(nums[0], nums[1]);

        for(int i = 2; i < nums.length; i++)
        {
            money[i] = Math.max(nums[i] + money[i - 2], money[i - 1]);
        }

        return money[nums.length - 1];

    }
    //505. The Maze II - Medium - Using Dijkstra and PriorityQueue
    public int shortestDistance(int[][] maze, int[] start, int[] destination) {
        int[][] distance = new int[maze.length][maze[0].length];
        for(int[] row: distance)
            Arrays.fill(row, Integer.MAX_VALUE);

        distance[start[0]][start[1]] = 0; //starting position
        int[][] dirs = {{0, 1}, {0, - 1}, {1, 0}, {-1, 0}};
        PriorityQueue<int[]> queue = new PriorityQueue<>((a,b) -> Integer.compare(a[2], b[2]));

        queue.offer(new int[]{start[0], start[1], 0});

        while(queue.size() > 0)
        {
            int[] curr = queue.poll();
            int row = curr[0];
            int col = curr[1];
            int dis = curr[2];

            if(distance[row][col] < dis)
                continue;

            for(int[] dir : dirs)
            {
                int r = row + dir[0];
                int c = col + dir[1];
                int count = 0;
                while(r >= 0 && r < maze.length && c >= 0 && c < maze[0].length && maze[r][c] == 0)
                {
                    r += dir[0];
                    c += dir[1];
                    count++;
                }

                if(distance[row][col] + count < distance[r - dir[0]][c - dir[1]])
                {
                    distance[r - dir[0]][c - dir[1]] = distance[row][col] + count;
                    queue.offer(new int[]{r - dir[0], c - dir[1], distance[r - dir[0]][c - dir[1]]});
                }
            }

        }

        return distance[destination[0]][destination[1]] == Integer.MAX_VALUE? -1 : distance[destination[0]][destination[1]];
    }
    //1514. Path with Maximum Probability - Medium - Use Dijkstra
    public double maxProbability(int n, int[][] edges, double[] succProb, int start, int end) {

        Map<Integer, List<ProbNode>> adjacencies = new HashMap<>();
        for(int i = 0; i < edges.length; i++)
        {
            int p1 = edges[i][0];
            int p2 = edges[i][1];
            double probabilities = succProb[i];

            adjacencies.putIfAbsent(p1, new ArrayList<>());
            adjacencies.get(p1).add(new ProbNode(p2, probabilities));

            adjacencies.computeIfAbsent(p2, x -> new ArrayList<>()).add(new ProbNode(p1, probabilities));
        }

        double[] probabilities = new double[n];
        Arrays.fill(probabilities, 0);

        //descending order
        Queue<ProbNode> pq = new PriorityQueue<ProbNode>((a, b) -> Double.compare(b.probility, a.probility));
        probabilities[start] = 1; //program still works without this, but will visited the start node again
        pq.add(new ProbNode(start, 1));

        while(pq.size() > 0)
        {
            ProbNode curr = pq.poll();
            int currNode = curr.node;

            double currNodeProbability = curr.probility;
            if(currNode == end)
                return currNodeProbability;

            if(adjacencies.containsKey(currNode))
            {
                for(ProbNode neighbor: adjacencies.get(currNode))
                {
                    if(probabilities[neighbor.node] < neighbor.probility * currNodeProbability)
                    {
                        probabilities[neighbor.node] = neighbor.probility * currNodeProbability;
                        pq.add(new ProbNode(neighbor.node, probabilities[neighbor.node]));
                    }
                }
            }
        }

        return 0;
    }

    public class ProbNode
    {
        int node;
        double probility;

        public ProbNode(int node, double probility)
        {
            this.node = node;
            this.probility = probility;
        }
    }

    //743. Network Delay Time - Medium - Using Dijkstra
    public int networkDelayTime(int[][] times, int n, int k) {
        Map<Integer, List<Pair<Integer, Integer>>> adjacencies = new HashMap<>();

        for(int[] time : times)
        {
            adjacencies.putIfAbsent(time[0], new ArrayList<>());
            adjacencies.get(time[0]).add(new Pair<>(time[1], time[2]));
        }

        int[] distances = new int[n + 1];
        Arrays.fill(distances, Integer.MAX_VALUE);

        dijkstra(adjacencies, distances, k, n);

        int answer = Integer.MIN_VALUE;
        for(int i = 1; i < distances.length; i++)
            answer = Math.max(answer, distances[i]);

        // INT_MAX signifies atleat one node is unreachable
        return answer == Integer.MAX_VALUE ? -1 : answer;
    }

    public void dijkstra(Map<Integer, List<Pair<Integer, Integer>>> adjacencies, int[] distances, int source, int n)
    {
        Queue<Pair<Integer, Integer>> pq = new PriorityQueue<>((a , b) -> a.getValue() - b.getValue());
        pq.add(new Pair<>(source, 0));

        //time for staring node is 0
        distances[source] = 0;

        while(pq.size() > 0)
        {
            Pair<Integer, Integer> curr = pq.remove();
            int currNode = curr.getKey();
            int currNodeTime = curr.getValue();

            if(currNodeTime > distances[currNode] || !adjacencies.containsKey(currNode))
                continue;

            for(Pair<Integer, Integer> edge : adjacencies.get(currNode))
            {
                int time = edge.getValue();
                int neighborNode = edge.getKey();

                if(distances[neighborNode] > time + currNodeTime)
                {
                    distances[neighborNode] = time + currNodeTime;
                    pq.add(new Pair<>(neighborNode, distances[neighborNode]));
                }
            }

        }
    }

    //1168. Optimize Water Distribution in a Village - Using Kruskal's Algo and Union Find
    public int minCostToSupplyWater(int n, int[] wells, int[][] pipes) {
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[2] - b[2]);
        int cost = 0;

        for(int i = 1; i <= n; i++)
        {
            pq.offer(new int[]{0, i, wells[i - 1]});
        }

        for(int[] pipe : pipes)
        {
            pq.offer(new int[]{pipe[0], pipe[1], pipe[2]});
        }

        UFR urf = new UFR(n + 1);
        while(pq.size() > 0)
        {
            int[] p = pq.poll();
            if(urf.find(p[0]) != urf.find(p[1]))
            {
                cost += p[2];
                urf.union(p[0], p[1]);
            }
        }
        return cost;
    }

    //1584. Min Cost to Connect All Points - Medium - Using prim
    public int minCostConnectPointsUsingPrimsAlgo(int[][] points) {
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[2] - b[2]);
        int cost = 0;
        int n = points.length;

        Set<Integer> visited = new HashSet<>(); //unvisited nodes
        visited.add(0);
        int[] startingPoint = new int[]{points[0][0], points[0][1]}; //pick the first one, but can be any point

        //find all the edges from the starting point;
        for(int j = 1; j < points.length; j++)
        {
            pq.offer(new int[]{0, j, distance(startingPoint, points[j])});
        }

        while(pq.size() > 0)
        {
            int[] p = pq.poll();
            if(!visited.contains(p[1]))
            {
                visited.add(p[1]);
                cost += p[2];
                for(int i = 0; i < n; i++)
                {
                    if(!visited.contains(i))
                    {
                        pq.offer(new int[]{p[2], i, distance(points[p[1]], points[i])});
                    }
                }
            }
            //terminate the loop once all nodes are connected
            if(visited.size() == n)
                break;
        }
        return cost;
    }

    //1584. Min Cost to Connect All Points - Medium
    public int minCostConnectPoints(int[][] points) {
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[2] - b[2]);
        int n = points.length - 1;

        for(int i = 0; i < n; i++)
        {
            for(int j = i + 1; j <= n; j++)
            {
                pq.offer(new int[]{i, j, distance(points[i], points[j])});
            }
        }

        UFR ufr = new UFR(points.length);
        int countEdges = 0;
        int weight = 0;

        while(pq.size() > 0)
        {
            int[] point = pq.poll();
            if(ufr.find(point[0]) != ufr.find(point[1]))
            {
                ufr.union(point[0], point[1]);
                weight += point[2];
                n--;
            }
            if(n == 0)
                break;
        }

        return weight;
    }

    public int distance(int[] x, int[] y)
    {
        return Math.abs(x[0] - y[0]) + Math.abs(x[1] - y[1]);
    }
}

