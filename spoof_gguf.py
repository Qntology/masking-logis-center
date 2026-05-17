import sys
from gguf import GGUFReader, GGUFWriter

def spoof_arch(input_path, output_path, old_arch, new_arch):
    print(f"Spoofing architecture {old_arch} -> {new_arch} in {input_path}")
    reader = GGUFReader(input_path)
    
    # We need to create a new writer and copy all KV pairs and tensors
    # but with updated names.
    writer = GGUFWriter(output_path, new_arch)
    
    # Copy and rename KV pairs
    for field in reader.fields.values():
        name = field.name
        if name == "general.architecture":
            continue # We set this in constructor
        
        # Replace prefix if it matches
        new_name = name.replace(f"{old_arch}.", f"{new_arch}.")
        
        # Handle array types and other types correctly
        if field.types:
            # This is complex to do manually with GGUFWriter for all types
            # but we can try to use add_key and add_val logic
            # However, GGUFWriter doesn't have a generic "add any type" method easily.
            pass

    # Actually, a simpler way is to just modify the file in place if possible
    # but GGUF format is not that easy to modify in place for strings of different lengths.
    
    # Let's try to use a more direct approach: 
    # Just update the GGUF constants in Python and run conversion as gemma4 
    # BUT with a hacked GGUFWriter that uses the name we want.
    
    # Actually, the user wants the names to be proper.
    # I'll just change the Python constants and run conversion again.
    pass

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: spoof.py <input> <output> <old_arch> <new_arch>")
    else:
        spoof_arch(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
