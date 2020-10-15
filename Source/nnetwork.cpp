
//#define USE_MPI

/*
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#ifdef USE_MPI
#include <mpi.h>
#endif
#include <chrono>
#include "deep_core.h"
#include "vector_ops.h"
#include "timer.h"


vector<string> split(const string &s, char delim) {
  stringstream ss(s);
  string item;
  vector<string> tokens;
  while (getline(ss, item, delim)) {
    tokens.push_back(item);
  }
  return tokens;
}

int main22(int argc, char * argv[]) {
  //double total_time = 0.0;
  Timing::Timer<Timing::Milliseconds> timer;

  string line;
  vector<string> line_v;
  int len, mpirank = 0;
  cout << "Loading data ...\n";
  vector<float> X_train;
  vector<float> y_train;
  ifstream myfile ("/home/joakim/Source/C++/Lab2_Code/train.txt");
  if (myfile.is_open())
  {
    while ( getline (myfile,line) )
    {
      line_v = split(line, '\t');
      int digit = strtof((line_v[0]).c_str(),0);
      for (unsigned i = 0; i < 10; ++i) {
        if (i == digit)
        {
          y_train.push_back(1.);
        }
        else y_train.push_back(0.);
      }

      int size = static_cast<int>(line_v.size());
      for (unsigned i = 1; i < size; ++i) {
        X_train.push_back(strtof((line_v[i]).c_str(),0));
      }
    }
    X_train = X_train/255.0;
    myfile.close();
  }

  else cout << "Unable to open file" << '\n';

  int xsize = static_cast<int>(X_train.size());
  int ysize = static_cast<int>(y_train.size());

  // Some hyperparameters for the NN
  int BATCH_SIZE = 256;
  float lr = .01/BATCH_SIZE;
  // Random initialization of the weights
  vector <float> W1 = random_vector(784*128);
  vector <float> W2 = random_vector(128*64);
  vector <float> W3 = random_vector(64*10);

#ifdef USE_MPI
  int mpisize;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

  MPI_Comm_rank(MPI_COMM_WORLD,&mpirank);
  BATCH_SIZE /= mpisize;
  // Seed the pseudo random generator differently for each rank
  srand (mpirank);
  // Gradient recv vectors and counts

  //initialize vectors dW1_recv of size (mpisize X size of W1)
  // do the same for dW2_recv and dW3_recv
  vector<float> dW1_recv(mpisize*W1.size(), 0);
  vector<float> dW2_recv(mpisize*W2.size(), 0);
  vector<float> dW3_recv(mpisize*W3.size(), 0);

  // You might need to initialize some variable depending on what MPI Call you are using to update vectors
#endif

  std::chrono::time_point<std::chrono::system_clock> t1,t2;
  cout << "Training the model ...\n";
  for (unsigned i = 0; i < 10000; ++i) {
    timer.start();
    // Building batches of input variables (X) and labels (y)
    int randindx = rand() % (42000-BATCH_SIZE);
    vector<float> b_X;
    vector<float> b_y;
    for (unsigned j = randindx*784; j < (randindx+BATCH_SIZE)*784; ++j){
      b_X.push_back(X_train[j]);
    }
    for (unsigned k = randindx*10; k < (randindx+BATCH_SIZE)*10; ++k){
      b_y.push_back(y_train[k]);
    }

    // Feed forward
    vector<float> a1 = relu(dot( b_X, W1, BATCH_SIZE, 784, 128 ));
    vector<float> a2 = relu(dot( a1, W2, BATCH_SIZE, 128, 64 ));
    vector<float> yhat = softmax(dot( a2, W3, BATCH_SIZE, 64, 10 ), 10);

    // Back propagation
    vector<float> dyhat = (yhat - b_y);
    // dW3 = a2.T * dyhat
    vector<float> dW3 = dot(transpose( &a2[0], BATCH_SIZE, 64 ), dyhat, 64, BATCH_SIZE, 10);
    // dz2 = dyhat * W3.T * relu'(a2)
    vector<float> dz2 = dot(dyhat, transpose( &W3[0], 64, 10 ), BATCH_SIZE, 10, 64) * reluPrime(a2);
    // dW2 = a1.T * dz2
    vector<float> dW2 = dot(transpose( &a1[0], BATCH_SIZE, 128 ), dz2, 128, BATCH_SIZE, 64);
    // dz1 = dz2 * W2.T * relu'(a1)
    vector<float> dz1 = dot(dz2, transpose( &W2[0], 128, 64 ), BATCH_SIZE, 64, 128) * reluPrime(a1);
    // dW1 = X.T * dz1
    vector<float> dW1 = dot(transpose( &b_X[0], BATCH_SIZE, 784 ), dz1, 784, BATCH_SIZE, 128);
    //printf(">>>>>>  BATCHSIZE = %d\n", BATCH_SIZE);

    // Updating the parameters << from own dWs >>
    W3 = W3 - lr * dW3;
    W2 = W2 - lr * dW2;
    W1 = W1 - lr * dW1;
#ifdef USE_MPI
    //since each process have updated Ws using their own dWs but that is not enough.
    // get the dWs from other processes using MPI collectives and do the updates for each dw recieved.
    MPI_Gather(dW1.data(), dW1.size(), MPI_FLOAT, dW1_recv.data(), dW1.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(dW2.data(), dW2.size(), MPI_FLOAT, dW2_recv.data(), dW2.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(dW3.data(), dW3.size(), MPI_FLOAT, dW3_recv.data(), dW3.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    if(mpirank == 0){
        for (int k = 0; k < mpisize; ++k) {
          if (k != mpirank) {
            //dws from process k
            // Updating the parameters
              vector<float> dW3_k(dW3_recv.begin()+k*dW3.size(), dW3_recv.begin()+((k+1)*dW3.size()));
              vector<float> dW2_k(dW2_recv.begin()+k*dW2.size(), dW2_recv.begin()+((k+1)*dW2.size()));
              vector<float> dW1_k(dW1_recv.begin()+k*dW1.size(), dW1_recv.begin()+((k+1)*dW1.size()));
              W3 = W3 - lr * dW3_k;
              W2 = W2 - lr * dW2_k;
              W1 = W1 - lr * dW1_k;
          }
        }
    }

#endif
    if ((mpirank == 0) && (i+1) % 100 == 0){
      cout << "Predictions:" << "\n";
      print ( yhat, 10, 10 );
      cout << "Ground truth:" << "\n";
      print ( b_y, 10, 10 );
      vector<float> loss_m = yhat - b_y;
      float loss = 0.0;
      for (unsigned k = 0; k < BATCH_SIZE*10; ++k){
        loss += loss_m[k]*loss_m[k];
      }
      timer.stop();
      cout << "Iteration #: "  << i << endl;
      cout << "Iteration Time: "  << timer.elapsed() << "s" << endl;
      cout << "Loss: " << loss/BATCH_SIZE << endl;
      cout << "*******************************************" << endl;
    };
  };
#ifdef USE_MPI
  MPI_Finalize();
#endif
  return 0;
}*/
