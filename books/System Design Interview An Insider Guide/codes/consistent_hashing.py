import hashlib
import bisect

class ConsistentHashing:
    def __init__(self, num_virtual_nodes=100):
        """
        Initialize the consistent hashing ring.
        :param num_virtual_nodes: Number of virtual nodes per real node (default 100).
        """
        self.num_virtual_nodes = num_virtual_nodes
        self.ring = {}  # Hash ring to store virtual nodes and their keys
        self.sorted_keys = []  # Sorted list of hash values (for binary search)
        self.nodes = set()  # Set of real nodes
    
    def _hash(self, key):
        """Hash a string to a 32-bit integer value using MD5."""
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)
    
    def _get_virtual_node_name(self, node): # Not necessary here. No special purpose to append ## behind virtual node name.
        """Generate virtual node names for the given real node."""
        return f"{node}##"  # Appending a unique suffix for each virtual node
    
    def add_node(self, node):
        """Add a new node to the ring."""
        self.nodes.add(node)
        
        # Add virtual nodes for the given real node
        for i in range(self.num_virtual_nodes):
            virtual_node = self._get_virtual_node_name(f"{node}-{i}")
            hash_value = self._hash(virtual_node)
            self.ring[hash_value] = node
            bisect.insort(self.sorted_keys, hash_value)
    
    def remove_node(self, node):
        """Remove a node from the ring."""
        self.nodes.remove(node)
        
        # Remove the virtual nodes associated with the real node
        for i in range(self.num_virtual_nodes):
            virtual_node = self._get_virtual_node_name(f"{node}-{i}")
            hash_value = self._hash(virtual_node)
            del self.ring[hash_value]
            self.sorted_keys.remove(hash_value)
    
    def get_node(self, key):
        """Get the node for a given key."""
        if not self.ring:
            return None
        
        # Hash the key to find its corresponding position on the ring
        key_hash = self._hash(key)
        
        # Use binary search to find the closest virtual node
        index = bisect.bisect(self.sorted_keys, key_hash)
        
        # If the index is equal to the length of the sorted keys, loop around
        if index == len(self.sorted_keys):
            index = 0
        
        # Find the corresponding node
        closest_hash = self.sorted_keys[index]
        return self.ring[closest_hash]

# Example usage
if __name__ == "__main__":
    # Initialize consistent hashing with 100 virtual nodes per real node
    ch = ConsistentHashing(num_virtual_nodes=100)

    # Add nodes to the ring
    ch.add_node("NodeA")
    ch.add_node("NodeB")
    ch.add_node("NodeC")

    # Sample data items (keys)
    keys = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]

    # Get the node for each key
    for key in keys:
        print(f"Key '{key}' is mapped to node: {ch.get_node(key)}")
    
    # Remove a node (e.g., "NodeB")
    print("\nRemoving NodeB from the ring...")
    ch.remove_node("NodeB")

    # Check the mapping again after node removal
    for key in keys:
        print(f"Key '{key}' is mapped to node: {ch.get_node(key)}")