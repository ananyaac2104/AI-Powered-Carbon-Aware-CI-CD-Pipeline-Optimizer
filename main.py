import sys
from src.core.decision_engine import DecisionEngine
from src.carbon_api_client import CarbonAPIClient

def main():
    # 1. Initialize Real Components
    engine = DecisionEngine()
    client = CarbonAPIClient()
    
    # 2. Fetch Real Carbon Intensity from API
    # (If API key is missing, it falls back to DB or 450)
    intensity = client.get_latest_intensity()
    
    # 3. Handle CLI Inputs or Fallbacks
    # Similarity: how similar the code is to previous versions (0.0 to 1.0)
    similarity = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85
    # Change Size: number of lines of code changed
    change_size = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    print(f"\n🌱 Green-Ops Analysis Starting...")
    print(f"   Carbon Intensity: {intensity if intensity else 'Fallback used'} gCO2/kWh")
    print(f"   Similarity:       {similarity}")
    print(f"   Change Size:      {change_size} LOC")

    # 4. Execute Dynamic Decision
    result = engine.decide(
        similarity=similarity,
        change_size=change_size,
        carbon_intensity=intensity
    )

    print("\n" + "="*40)
    print("      🟢 GREEN-OPS FINAL DECISION 🟢")
    print("="*40)
    print(f"   Decision:    {result['decision']}")
    print(f"   Reason:      {result['reason']}")
    print(f"   Probability: {result['probability']}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()