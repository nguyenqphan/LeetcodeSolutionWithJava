import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Set;

public class Main {

    public static void main(String[] args) {
	// write your code here
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
