import sys
import numpy as np
from tensorflow import keras
#import matplotlib.pyplot as plt
import tensorflow as tf
import cantera as ct
from scipy import optimize as opt
import numpy as np
import itertools

def states_new_init(A):
    R_new = ct.Reaction.fromCti('''reaction('O2 + 2 H2 => 2 H2O',
            [%e, 0.0, 0.0])'''%(A))
    #print(type(R_new))
    #print(type(gas.reactions()))
    gas2 = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
    species=gas.species(), reactions=[R_new])
    gas2.TPX = initial_state
    r_new = ct.IdealGasConstPressureReactor(gas2, energy = 'off')
    t_new = 0.0
    states_new = ct.SolutionArray(gas2, extra=['t'])
    sim_new = ct.ReactorNet([r_new])
    tt = []
    TT = []
    for n in range(100):
        '''t_new += 1.e-5
        sim_new.advance(t_new)
        #print(t_new)
        tt.append(1000 * t_new*1000)
        TT.append(r_new.T)'''
        t_new += 1.e-5
        sim_new.advance(t_new)
        states_new.append(r_new.thermo.state, t=t_new*1e3)
    return states_new, gas2



def obj_func(A,states_ref):
    ret = 0.
    states_new,gas2 = states_new_init(A)
    for n in range(100):
        ret += (states_new.X[n,gas2.species_index('H2')] - states_ref.X[n,gas2.species_index('H2')])**2/100
    return ret
#return abs(a-b)


gas = ct.Solution('gri30.xml')
initial_state = 1500, ct.one_atm, 'H2:2,O2:1'
gas.TPX = 1500.0, ct.one_atm, 'H2:2,O2:1'
r = ct.IdealGasConstPressureReactor(gas,energy = 'off')
sim = ct.ReactorNet([r])
time = 0.0
states = ct.SolutionArray(gas, extra=['t'])

#
i = (j for j in range(0,10))
N = (j for j in range(1,11))
#
for n in range(100):
    time += 1.e-5
    sim.advance(time)
    states.append(r.thermo.state, t=time*1e3)
    #print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T,
    #                                       r.thermo.P, r.thermo.u))

#min = opt.minimize(lambda x: obj_func(x,states),1000.,method='Nelder-Mead')
#print(min)

min = opt.minimize(lambda x: obj_func(x,states),1000.,method='Nelder-Mead')

states_new,gas2 = states_new_init(min.x[0])




#model init
model = keras.Sequential()

model.add(keras.layers.Dense(60, activation='elu',input_shape=(1,)))
#model.add(keras.layers.Dense(1, activation='elu'))
model.add(keras.layers.Dense(60, activation='softmax'))
#model.add(keras.layers.Dense(1, activation='elu'))
model.add(keras.layers.Dense(60, activation='elu'))
#model.add(keras.layers.Dense(64, activation='sigmoid'))
#model.add(keras.layers.Dense(1, activation='sigmoid'))
model.add(keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l1(0.01)))



#add parameters of model
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='MSLE',
              metrics=['mae'])


#add dataset
arren_array = np.random.normal(min.x[0],10e2,200)
states_new_arr = []
gas_new =  []

for i in range(len(arren_array)):
    value_1, value_2 = states_new_init(arren_array[i])
    states_new_arr.append(value_1)
    gas_new.append(value_2)

#set_train_data

xtrain = np.array([np.array([i],dtype='float32') for i in arren_array[:100]])
ytrain = np.array([np.array([obj_func(i,states)],dtype='float32') for i in arren_array[:100]])

#set_test_data
xtest = np.array([np.array([i],dtype='float32') for i in arren_array[100:]])
ytest = np.array([np.array([obj_func(i,states)],dtype='float32') for i in arren_array[100:]])

#create_train_dataset
dataset = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
dataset = dataset.batch(64).repeat()
#create_test_dataset
val_dataset = tf.data.Dataset.from_tensor_slices((xtest, ytest))
val_dataset = val_dataset.batch(64).repeat()

#train_the_model
model.fit(dataset, epochs=300, steps_per_epoch=100,
          validation_data=val_dataset,
          validation_steps=10)

#predictions

arren_array_pred = np.random.normal(min.x[0],10,10)
states_new_arr_pred = []
gas_new_pred =  []

for i in range(len(arren_array)):
    value_1, value_2 = states_new_init(arren_array[i])
    states_new_arr_pred.append(value_1)
    gas_new_pred.append(value_2)


xpred = np.array([np.array([i],dtype='float32') for i in arren_array_pred])

predict_results = model.predict(xpred, steps=30)

predictions = list(itertools.islice(predict_results,len(xpred)))


#output predictions
for k,val in enumerate(predictions):
  print(str(xpred[k]) + ' ' + str(val))


'''
C = plt.cm.winter(np.linspace(0,1,10))

plt.clf()
plt.subplot(2, 2, 1)
plt.plot(states.t, states.T,color='red')
plt.plot(states_new.t, states_new.T,color='green')
plt.xlabel('Time (ms)')
plt.ylabel('Temperature (K)')
plt.subplot(2, 2, 2)
plt.plot(states.t, states.X[:,gas.species_index('O2')],color='red')
plt.plot(states_new.t, states_new.X[:,gas2.species_index('O2')],color='green')
plt.xlabel('Time (ms)')
plt.ylabel('O2 Mole Fraction')
plt.subplot(2, 2, 3)
plt.plot(states.t, states.X[:,gas.species_index('H2O')],color='red')
plt.plot(states_new.t, states_new.X[:,gas2.species_index('H2O')],color='green')
plt.xlabel('Time (ms)')
plt.ylabel('H2O Mole Fraction')
plt.subplot(2, 2, 4)
plt.plot(states.t, states.X[:,gas.species_index('H2')],color='red')
plt.plot(states_new.t, states_new.X[:,gas2.species_index('H2')],color='green')
plt.xlabel('Time (ms)')
plt.ylabel('H2 Mole Fraction')
plt.tight_layout()
plt.show()
'''
