#include "nccl.h"
#include "mpi.h"
#include "cuda.h"
#include <iostream>
#include <memory>
#include <thread>

constexpr size_t bsize = 4;

int main(int argc, char** argv) {
  int size, rank;
  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  cudaSetDevice(rank % 4);
  
  cudaStream_t s;
  cudaStreamCreate(&s);  

  ncclComm_t comm;
  ncclUniqueId uid;
  if(rank == 0){
    ncclGetUniqueId(&uid);
  }
  MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclCommInitRank(&comm, size, uid, rank);

  auto data = std::make_unique<float[]>(bsize);
  auto recvbuf = std::make_unique<float[]>(bsize);
  for(int i = 0; i < bsize; i++){
    data[i] = (float) rank + i;
  }
  
  ncclAllReduce(data.get(), recvbuf.get(), bsize, ncclFloat, ncclSum, comm, s);

  cudaStreamSynchronize(s);

  std::cout << "Done " << data[0] << "\n";
  
  std::cout << "Sleeping \n";

  std::this_thread::sleep_for(std::chrono::seconds(120));
  
  std::cout << "Awake Sleeping \n";

  
  cudaStreamDestroy(s);
  ncclCommDestroy(comm);
  MPI_Finalize();
}
