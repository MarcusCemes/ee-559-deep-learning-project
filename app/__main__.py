if __name__ == "__main__":
    print("Initialising...")

    from asyncio import run
    from .app import main

    run(main())
