import tensorflow as tf
import itertools
import math

xtrain = [i for i in range(1,21,2)]
ytrain = [math.exp(i) for i in range(1,21,2)]
xtest = [22,24,26,29,30,34]
##xtrain2 = [i for i in range(200,300)]
##ytrain2 = [i**2 for i in range(200,300)]
xpred = [i for i in range(2,21,2)]
ytest = [math.exp(val) for val in xtest]

#xtrain = [1,2,3,4,5,6,7,8,9,10]
#ytrain = [1,4,9,16,25,36,49,64,81,100]
#xtest = [11,12,13,14,15,16,17,18,19,20]
#xtrain2 = [1,1.5,1.9,3,5,7,9,11,13,15]
#ytrain2 = [1,2.25,3.61,9,25,49,81,121,169,225]
#xpred = [2,4,6,8,10,12,14,16,1.5]
#ytest = [11**2,12**2,13**2,14**2,15**2,16**2,17**2,18**2,19**2,20**2]




feature_x = tf.feature_column.numeric_column(key='xval')
#feature_y = tf.feature_column.numeric_column(key='yval')

#inputs = tf.data.Dataset((xtrain,ytrain))

model_new = tf.estimator.DNNRegressor(hidden_units=[25,25],feature_columns=[feature_x])
''', optimizer=lambda: tf.train.AdamOptimizer(
        learning_rate=tf.train.exponential_decay(
            learning_rate=0.01,
            global_step=tf.train.get_global_step(),
            decay_steps=10000,
            decay_rate=0.96)))
'''
def input_fn_train():
  return ({'xval':xtrain},ytrain)
def input_fn_eval():
  return ({'xval':xtest},ytest)

def input_fn_predict():
  return ({'xval':xpred},0)


model_new.train(input_fn=input_fn_train,steps=1000)
eval_result = model_new.evaluate(input_fn=input_fn_eval,steps=1000)
average_loss = eval_result["average_loss"]
predict_result = model_new.predict(input_fn=input_fn_predict,yield_single_examples=True)
predictions = list(itertools.islice(predict_result,len(xpred)))
#print('Predictions: {}'.format(str(predictions)))
#[predict_result.__next__() for i in range(100)]
for k,val in enumerate(predictions):
  print(str(xpred[k]) + ' ' + str(val['predictions']) + ' ' + str(100*abs(1 - val['predictions'][0]/math.exp(xpred[k]))))
#
#
#print("eval  result:")
#print(eval_result)
##print("predict result:")
##print(predict_result)
## Convert MSE to Root Mean Square Error (RMSE).
#print("\n" + 80 * "*")
#print("\nRMS error for the test set: {:.0f}%"
#      .format( average_loss/100.0))
#
#print()
