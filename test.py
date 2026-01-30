# from paddlex import create_model
# model = create_model(model_name="PP-LCNet_x0_25_textline_ori")
# output = model.predict("test\\180.jpg",  batch_size=1)
# for res in output:
#     res.print(json_format=False)
#     res.save_to_img("./output/demo.png")
#     res.save_to_json("./output/res.json")
from paddlex import create_model
model = create_model(model_name="PP-OCRv5_server_rec")
output = model.predict(input="test\\901.jpg", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output2/")
    res.save_to_json(save_path="./output2/res.json")