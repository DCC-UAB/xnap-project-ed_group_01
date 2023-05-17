from phocnet_utils import *
from punet_utils import *

# Testing different function from the PHOCnet repo

# get_most_common_n_grams -> dona els N-grames més coumns, els podem passar com a paràmetre 
text_lorem = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum"
text_list = text_lorem.split(" ")
#print(text_list)
res = get_most_common_n_grams(text_list)
#print(res)

uni_grames = get_most_common_n_grams(text_list, 50, 1)
bi_grames = get_most_common_n_grams(text_list, 50, 2)
print(len('abcdefghijklmnopqrstuvwxyz'))
print(len(bi_grames))
res_phoc = build_phoc(["hola"] , 'abcdefghijklmnopqrstuvwxyz', [1], [1,2], bi_grames)
print(res_phoc)
print(res_phoc.shape)
#print(list(bi_grames.keys()))
#res_punet = generate_label("hola")
#print(res_punet)