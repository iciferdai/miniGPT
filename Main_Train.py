from modelMgmt import *

def pre_init(need_data=True):
    print('init model...')
    my_model = MiniGPT()
    my_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if need_data:
        print('Preparing data...')
        train_dataloader = process_data()
    else:
        train_dataloader = None
    print('init ModelManagement...')
    my_mgmt = ModelManagement(my_model, train_dataloader, my_dev)
    return my_mgmt

def main_train(steps, fp):
    my_mgmt = pre_init()
    print('init train...')
    my_mgmt.init_train()
    my_mgmt.init_weights()
    print('Start train...')
    my_mgmt.train_steps(steps, fp)
    my_mgmt.save_state()
    my_mgmt.show_dashboard()

def load_train(steps, cpt_name, sts_name, fp):
    my_mgmt = pre_init()
    print('init train...')
    my_mgmt.init_train()
    my_mgmt.load_checkpoint(cpt_name)
    my_mgmt.load_state(sts_name)
    print('Start train...')
    my_mgmt.train_steps(steps, fp)
    my_mgmt.save_state()
    my_mgmt.show_dashboard()

def check_status(cpt_name, sts_name):
    my_mgmt = pre_init(False)
    print('load status of best_test...')
    #my_mgmt.init_train()
    my_mgmt.init_dashboard()
    #my_mgmt.load_checkpoint(cpt_name)
    my_mgmt.load_state(sts_name)
    my_mgmt.progress_info(True)
    my_mgmt.show_dashboard()

if __name__ == '__main__':
    #main_train(30000, True)
    #load_train(20000,'CheckPoint_Ep40000_1.4955.pth','State_Ep40000_1.4365.pkl', True)
    check_status('CheckPoint_Ep60000_1.4640.pth','State_Ep60000_1.4241.pkl')