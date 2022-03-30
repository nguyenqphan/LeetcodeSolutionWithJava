import javafx.util.Pair;
import java.util.*;

import static java.util.List.*;

public class Main {
    public static void main(String[] args) {}

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
    //18. 4sum - Medium
    public static List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<>();
        int size = nums.length - 1;
        if(nums.length < 4)
            return res;

        Arrays.sort(nums);
        for(int i = 0; i <= size - 3; i++)
        {
            if(i != 0 && nums[i] == nums[i - 1])
                continue;

            for(int j = i + 1; j <= size - 2; j++)
            {
                if(j != i + 1 && nums[j] == nums[j - 1])
                    continue;

                int left = j + 1;
                int right = size;
                while(left < right)
                {
                    int sum = nums[i] + nums[j] + nums[left] + nums[right];
                    if(sum == target)
                    {
                        List<Integer> ans = Arrays.asList(nums[i], nums[j], nums[left++], nums[right--]);
                        res.add(ans);
                        while(left < right && nums[left] == nums[left - 1])
                            left++;
                    }else if(sum < target)
                    {
                        left++;
                    }else
                    {
                        right--;
                    }

                }
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
                n--;
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

