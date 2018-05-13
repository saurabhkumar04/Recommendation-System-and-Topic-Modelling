import tensorflow as tf

def build__fact_mach_tf_graph_class(row_count, features, rank, 
                   optimizer = tf.train.GradientDescentOptimizer(.01),
                   l2_scale = .005 ):
    tf.reset_default_graph()
#    features=5
#    classes=2
#    rank=10
#    x = tf.placeholder(tf.float32, [None, features]) # mnist data image of shape 28*28=784
    y = tf.placeholder('float', shape=[None, 1])
    x = tf.placeholder(tf.float32, [None, features])
#    x = tf.sparse_placeholder(dtype=tf.float32, shape=[None, features])
    # Set model weights
    
    W = tf.Variable(tf.random_normal([features],stddev = .01))
    b = tf.Variable(tf.random_normal([1],stddev = .01))
    
    # interaction factors, randomly initialized 
    V = tf.Variable(tf.random_normal([rank, features], stddev=0.01))
    if rank==0:
        interactions=0
    else:
        interactions = (tf.multiply(0.5,
                        tf.reduce_sum(
                            tf.subtract(
                                tf.pow( tf.matmul(x, tf.transpose(V),a_is_sparse=True), 2),
                                tf.matmul(tf.pow(x, 2), tf.transpose(tf.pow(V, 2)))),
                            1, keep_dims=True)))
        
    l2_regularizer = tf.contrib.layers.l2_regularizer(
        scale=l2_scale, scope=None
    )
    weights = tf.trainable_variables() # all vars of your graph
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
    
    
    # Construct model
    
    pred = np.add(tf.add(b, tf.reduce_sum(tf.multiply(W,x),axis=1, keep_dims=True)),interactions)
    # Minimize error using cross entropy
    mse = tf.reduce_mean(tf.square(tf.subtract(y, pred)))
    loss = mse+regularization_penalty
    optimizer = optimizer.minimize(loss)
    # Calculate accuracy for 3000 examples
    
    accuracy = tf.reduce_mean((tf.subtract(y, pred)))
    return [pred,optimizer, accuracy, x,y, V]

def fact_machine_fit(X_train, y_train, rank, display_step = 1,
                     training_epochs = 4, batch_size=1024,
                     optimizer=tf.train.GradientDescentOptimizer(.01),
                     l2_scale=.005,var1=[]):
    X_train = X_train.astype('float')
    Y_train = y_train.astype('float')
    
    if len(var1)==2:
        X_test = var1[0].astype('float')
        Y_test = var1[1].astype('float')
    pred,optimizer, accuracy, x,y, V = build__fact_mach_tf_graph_class(Y_train.shape[0],
                                                          X_train.shape[1],
                                                          rank,
                                                          optimizer = optimizer,
                                                          l2_scale = l2_scale
                                                          )
    init = tf.global_variables_initializer()

    predictions = 0
    # Start training
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(Y_train.shape[0]/batch_size)
        rand_ind = np.random.randint(Y_train.shape[0], size = Y_train.shape[0])
        # Loop over all batches
        for i in range(0,int(Y_train.shape[0]/batch_size)):
            batch_xs = X_train[rand_ind][i*batch_size:(i+1)*batch_size]
            batch_ys = Y_train[rand_ind][i*batch_size:(i+1)*batch_size]
#            batch_xs = tf.SparseTensorValue(
#                    indices=np.array([batch_xs.row, batch_xs.col]).T,
#                    values=batch_xs.data,
#                    dense_shape=batch_xs.shape)
        # Loop over all batches
#            print(sess.run(V))
            # Fit training using batch data
            _, c = sess.run([optimizer, accuracy], feed_dict={x: batch_xs.todense(),
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            if len(var1)==2:
#                X_test = tf.SparseTensorValue(
#                    indices=np.array([X_test.row, X_test.col]).T,
#                    values=X_test.data,
#                    dense_shape=X_test.shape)
#                print(np.mean(np.sum(sess.run(pred,feed_dict={x: X_test})*Y_test,1)))
                print(sess.run(accuracy,feed_dict={x: X_test.todense(),y: Y_test}))
    print ("Optimization Finished!")
    return [sess, pred, accuracy, x, y]


test = dt_rest[dt_rest.date>='2016-06']

train = dt_rest[dt_rest.date<'2016-06']

np.sum(pd.merge(train[['user_id','business_id']].groupby('user_id').count().reset_index(),test[['user_id','business_id']].groupby('user_id').count().reset_index(), on='user_id').business_id_y)

from sklearn import preprocessing
lb_usr = preprocessing.LabelBinarizer()
lb_usr = lb_usr.fit(train.user_id)
usr_tr = sparse.csr_matrix(lb_usr.transform(train.user_id))

usr_ts = sparse.csr_matrix(lb_usr.transform(test.user_id))

lb_bis = preprocessing.LabelBinarizer()
lb_bis = lb_bis.fit(train.business_id)
bis_tr = sparse.csr_matrix(lb_bis.transform(train.business_id))

bis_ts = sparse.csr_matrix(lb_bis.transform(test.business_id))

X_train = sparse.hstack([usr_tr, bis_tr])
X_ts = sparse.hstack([usr_ts, bis_ts])

y_ts = test.stars_y.apply(lambda a:1.0 if a>=4 else 0)
y_tr = train.stars_y.apply(lambda a:1.0 if a>=4 else 0)
y_tr = np.array([a for a in y_tr.apply(lambda a:([a,1-a]))])
y_ts = np.array([a for a in y_ts.apply(lambda a:([a,1-a]))])

model=fact_machine_fit(X_train.tocsr(),np.expand_dims(train.stars_y,axis=1),20,training_epochs =10,var1=[X_ts.tocsr()[1:10000],np.expand_dims(test.stars_y,axis=1)[1:10000]])

model[0].run(model[1],feed_dict={model[3]: X_ts.tocsr()[100:120].todense(),model[4]: np.expand_dims(test.stars_y,axis=1)[100:120]})

model=fact_machine_fit(X_train.tocsr(),y_tr,20,l2_scale=.005,optimizer=tf.train.AdamOptimizer(.01),training_epochs =10,var1=[X_ts.tocsr()[1:10000],y_ts[1:10000]])


test = dt_rest[dt_rest.date>='2016-06']

train = dt_rest[dt_rest.date<'2016-06']

np.sum(pd.merge(train[['user_id','business_id']].groupby('user_id').count().reset_index(),test[['user_id','business_id']].groupby('user_id').count().reset_index(), on='user_id').business_id_y)

from sklearn import preprocessing
lb_usr = preprocessing.LabelBinarizer()
lb_usr = lb_usr.fit(train.user_id)
usr_tr = sparse.csr_matrix(lb_usr.transform(train.user_id))

usr_ts = sparse.csr_matrix(lb_usr.transform(test.user_id))

lb_bis = preprocessing.LabelBinarizer()
lb_bis = lb_bis.fit(train.business_id)
bis_tr = sparse.csr_matrix(lb_bis.transform(train.business_id))

bis_ts = sparse.csr_matrix(lb_bis.transform(test.business_id))

X_train = sparse.hstack([usr_tr, bis_tr])
X_ts = sparse.hstack([usr_ts, bis_ts])

y_ts = test.stars_y.apply(lambda a:1.0 if a>=4 else 0)
y_tr = train.stars_y.apply(lambda a:1.0 if a>=4 else 0)
y_tr = np.array([a for a in y_tr.apply(lambda a:([a,1-a]))])
y_ts = np.array([a for a in y_ts.apply(lambda a:([a,1-a]))])

model=fact_machine_fit(X_train.tocsr(),np.expand_dims(train.stars_y,axis=1),20,training_epochs =10,var1=[X_ts.tocsr()[1:10000],np.expand_dims(test.stars_y,axis=1)[1:10000]])

model[0].run(model[1],feed_dict={model[3]: X_ts.tocsr()[100:120].todense(),model[4]: np.expand_dims(test.stars_y,axis=1)[100:120]})

model=fact_machine_fit(X_train.tocsr(),y_tr,20,l2_scale=.005,optimizer=tf.train.AdamOptimizer(.01),training_epochs =10,var1=[X_ts.tocsr()[1:10000],y_ts[1:10000]])



