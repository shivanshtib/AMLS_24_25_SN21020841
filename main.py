import sys
from pathlib import Path

# Set directory paths
TASK_A_DIR = Path(__file__).resolve().parent / "A"
TASK_B_DIR = Path(__file__).resolve().parent / "B"

# Append the Task_A and Task_B directories relative to the current script
sys.path.append(str(TASK_A_DIR))
sys.path.append(str(TASK_B_DIR))

# Import the main functions from the task modules
import task_a_test as ta_te
import task_b_test as tb_te
import task_a_train as ta_tr
import task_b_train as tb_tr


def main():
    while True:
        print("\n=== Menu ===")
        print("1. Task A : Test Model")
        print("2. Task A : Visualise Model Training")
        print("3. Task B : Test Model")
        print("4. Task B : Visualise Model Training")
        print("5. Exit")
        
        try:
            # Get user input
            choice = int(input("Enter your choice (1-5): "))
            
            if choice == 1:
                print("\nUsing best_model_A_final...")
                print("Starting Test on Task A...\n")
                ta_te.run()
                print("Testing completed.\n")
            elif choice == 2:
                print("\nStarting Training on Task A...\n")
                ta_tr.run()
                print("\nTraining completed.\n")
            elif choice == 3:
                print("\nUsing best_model_B_final...")
                print("Starting Test on Task B...\n")
                tb_te.run()
                print("Testing completed.\n")
            elif choice == 4:
                print("\nStarting Training on Task B...\n")
                tb_tr.run()
                print("\nTraining completed.\n")
            elif choice == 5:
                print("\nExiting. Goodbye!\n")
                break
            else:
                print("\nInvalid choice. Please enter a number between 1 and 5.")
        except ValueError:
            print("\nInvalid input. Please enter a valid number.")

if __name__ == "__main__":
    main()
