public class UFR {
    public int[] parent;
    public int[] rank;

    public UFR(int size)
    {
        parent = new int[size];
        rank = new int[size];
        for(int i = 0; i < size; i++)
        {
            parent[i] = i;
            rank[i] = 1;
        }
    }

    public int find(int x)
    {
        if(parent[x] == x)
            return x;
        else
            return parent[x] = find(parent[x]);
    }

    public void union(int x, int y)
    {
        int rootX = find(x);
        int rootY = find(y);

        if(rootX != rootY)
        {
            if(rank[rootX] > rank[rootY])
            {
                parent[rootY] = rootX;
            }
            else if (rank[rootX] < rank[rootY])
            {
                parent[rootX] = rootY;
            }else
            {
                parent[rootX] = rootY;
                rank[rootY]++;
            }
        }
    }
}
