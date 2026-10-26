import sys


def main():
    try:
        from .rag_pipeline import RagPipeline
        import fire
        fire.Fire(RagPipeline)
    except ImportError as e:
        print(f"Error importing fire module: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error initializing CLI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
