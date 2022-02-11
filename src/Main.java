import javafx.util.Pair;
import java.util.*;
public class Main {

    public static void main(String[] args) {
	// write your code here
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
