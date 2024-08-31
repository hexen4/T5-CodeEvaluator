from datasets import load_dataset

if __name__ == "__main__":
    parent_directory = r"C:\Users\jlhb83\Desktop\Python Projects\hexen"
    ds = load_dataset("codeparrot/codeparrot-clean")
    df = load_dataset("nampdn-ai/tiny-codes")
    ds.save_to_disk(f"{parent_directory}/codeparrot-clean")
    df.save_to_disk(f"{parent_directory}/tiny-codes")




