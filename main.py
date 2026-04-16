from src.core.decision_engine import DecisionEngine

def main():
    engine = DecisionEngine()

    result = engine.decide(
        similarity=0.6,
        change_size=25,
        carbon_intensity=350
    )

    print("\n=== FINAL DECISION ===")
    print(result)

if __name__ == "__main__":
    main()