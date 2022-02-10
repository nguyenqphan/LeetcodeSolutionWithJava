import javafx.util.Pair;
import java.util.*;
public class Main {

    public static void main(String[] args) {
	// write your code here
    }

    //1514. Path with Maximum Probability - Medium - Use Dijkstra
    public double maxProbability(int n, int[][] edges, double[] succProb, int start, int end) {

        Map<Integer, List<ProbNode>> adjacencies = new HashMap<>();
        for(int i = 0; i < edges.length; i++)
        {
            int p1 = edges[i][0];
            int p2 = edges[i][1];
            double probility = succProb[i];

            adjacencies.putIfAbsent(p1, new ArrayList<>());
            adjacencies.get(p1).add(new ProbNode(p2, probility));

            adjacencies.computeIfAbsent(p2, x -> new ArrayList<>()).add(new ProbNode(p1, probility));
        }

        double[] probilities = new double[n];
        Arrays.fill(probilities, 0);

        //descending order
        Queue<ProbNode> pq = new PriorityQueue<ProbNode>((a, b) -> Double.compare(b.probility, a.probility));
        probilities[start] = 1; //program still works without this, but will visited the start node again
        pq.add(new ProbNode(start, 1));

        while(pq.size() > 0)
        {
            ProbNode curr = pq.poll();
            int currNode = curr.node;

            double currNodeProbility = curr.probility;
            if(currNode == end)
                return currNodeProbility;

            if(adjacencies.containsKey(currNode))
            {
                for(ProbNode neighbor: adjacencies.get(currNode))
                {
                    if(probilities[neighbor.node] < neighbor.probility * currNodeProbility)
                    {
                        probilities[neighbor.node] = neighbor.probility * currNodeProbility;
                        pq.add(new ProbNode(neighbor.node, probilities[neighbor.node]));
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
