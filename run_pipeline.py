from clean_data import main as clean_data_main
from evaluate_model import main as evaluate_main
from feature_selection import main as feature_select_main
from train_model import main as train_main


def main():
    clean_data_main()
    feature_select_main()
    train_main()
    evaluate_main()


if __name__ == "__main__":
    main()
