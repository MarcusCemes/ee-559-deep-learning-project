if __name__ == "__main__":
    print("⏳ Initialising...")

    from asyncio import run

    try:
        from .app import main

    except ModuleNotFoundError as e:
        print(f"⚠️ {e}")
        print("❓ Did you forget to activate the virtual environment?")
        exit(1)

    run(main())
