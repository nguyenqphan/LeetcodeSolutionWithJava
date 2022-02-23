import java.util.HashMap;
import java.util.Map;

public class TrieNode {
    boolean isWord;
    Map<Character, TrieNode> children;

    public TrieNode() {
        isWord = false;
        children = new HashMap<Character, TrieNode>();
    }
}

