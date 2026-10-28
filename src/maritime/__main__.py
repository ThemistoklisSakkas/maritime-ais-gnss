from maritime.cli import main

# The guard matters on Windows: process pools (B5) re-import the main module in every
# worker, and an unguarded main() would start the whole study again in each of them.
if __name__ == "__main__":
    main()
