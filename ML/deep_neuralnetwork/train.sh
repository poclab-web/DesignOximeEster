python_path="path/to/python"
epoch=1000
batch_size=128
log_path="./log/T1"
df_path="dataset csv path"
learning_rate=0.01
y_column="T1"
normalize=True
delete_logs=True
$python_path main.py --epoch $epoch --batch_size $batch_size --log_path $log_path --df_path $df_path --learning_rate $learning_rate --y_column $y_column --normalize $normalize --delete_logs $delete_logs