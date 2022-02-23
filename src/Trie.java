//208. Implement Trie (Prefix Tree) - Medium
public class Trie {
    TrieNode root;

    public Trie() {
        root = new TrieNode();
    }

    public void insert(String word) {
        TrieNode curr = root;
        for (int i = 0; i < word.length(); i++) {
            Character key = word.charAt(i);
            if (curr.children.get(key) == null) {
                curr.children.put(key, new TrieNode());
            }
            curr = curr.children.get(key);
        }

        curr.isWord = true;
    }

    public boolean search(String word) {
        TrieNode curr = root;

        for (int i = 0; i < word.length(); i++) {
            Character key = word.charAt(i);
            if (curr.children.get(key) == null)
                return false;

            curr = curr.children.get(key);
        }

        return curr.isWord;
    }

    public boolean startsWith(String prefix) {
        TrieNode curr = root;

        for (int i = 0; i < prefix.length(); i++) {
            Character key = prefix.charAt(i);
            if (curr.children.get(key) == null)
                return false;

            curr = curr.children.get(key);
        }

        return true;
    }
}
