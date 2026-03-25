from modelMgmt import *
from Main_Train import pre_init

def main_eval_manual(cpt_name):
    my_mgmt = pre_init(False)
    print('init evaluate...')
    my_mgmt.init_eval()
    print('load checkpoint...')
    my_mgmt.load_checkpoint(cpt_name, True)
    time.sleep(0.01)
    input_t = input("\nPress send input: ")
    my_mgmt.predict_manual(input_t)
    my_mgmt.predict_manual(input_t,'TOP_K',0.5, 3)
    my_mgmt.predict_manual(input_t,'TOP_P',0.8, 0.9)

if __name__ == '__main__':
    main_eval_manual('best_loss_cpt.pth')