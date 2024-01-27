#include <mpi.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace fs = std::filesystem;

constexpr size_t BLOCK_SIZE = 1024 * 1024;

void calculate_checksum(uint8_t *data, size_t len, uint8_t *obuf, int rank, int num_procs) {
    int num_block = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint8_t prev_md[SHA512_DIGEST_LENGTH];

    int blocks_per_process = (num_block + num_procs - 1) / num_procs;
    int start_block = rank * blocks_per_process;
    int end_block = std::min(num_block, (rank + 1) * blocks_per_process);

    if (rank == 0) {
        SHA512(nullptr, 0, prev_md);
    } else {
        MPI_Recv(prev_md, SHA512_DIGEST_LENGTH, MPI_UNSIGNED_CHAR, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for (int i = start_block; i < end_block; i++) {
        uint8_t buffer[BLOCK_SIZE + SHA512_DIGEST_LENGTH]{};
        size_t current_block_size = std::min(BLOCK_SIZE, len - i * BLOCK_SIZE);

        // 复制当前块的数据
        memcpy(buffer, data + i * BLOCK_SIZE, current_block_size);
        // 在当前块的末尾连接上第i-1个块的校验码
        memcpy(buffer + current_block_size, prev_md, SHA512_DIGEST_LENGTH);

        // 对连接后的数据进行SHA512校验，得到第i个块的校验码
        SHA512(buffer, current_block_size + SHA512_DIGEST_LENGTH, obuf + i * SHA512_DIGEST_LENGTH);

        // 更新prev_md为当前块的校验码，供下一个块使用
        memcpy(prev_md, obuf + i * SHA512_DIGEST_LENGTH, SHA512_DIGEST_LENGTH);
    }
    if (rank < num_procs - 1) {
        MPI_Send(prev_md, SHA512_DIGEST_LENGTH, MPI_UNSIGNED_CHAR, rank + 1, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // 将进程数量调整为8
    num_procs = 8;

    if (rank == 0) {
        if (argc < 2) {
            std::cout << "Usage: " << argv[0] << " <input_file>" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // 读取文件
        std::ifstream input_file(argv[1], std::ios::binary | std::ios::ate);
        std::size_t file_size = input_file.tellg();
        input_file.seekg(0, std::ios::beg);

        uint8_t *file_data = new uint8_t[file_size];
        input_file.read(reinterpret_cast<char *>(file_data), file_size);

        // 将文件划分成大小为 BLOCK_SIZE 的块
        int num_blocks = (file_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint8_t *checksums = new uint8_t[num_blocks * SHA512_DIGEST_LENGTH]();

        // 计算第 -1 块的校验码（空文件的校验码）
        SHA512(nullptr, 0, checksums - SHA512_DIGEST_LENGTH);

        // 分发文件大小和数据
        MPI_Bcast(&file_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(file_data, file_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        // 计算主进程分配的部分数据的校验和
        calculate_checksum(file_data, file_size, checksums, rank, num_procs);

        // 收集其他进程的校验和结果
        for (int i = 1; i < num_procs; i++) {
            MPI_Recv(checksums + i * SHA512_DIGEST_LENGTH, SHA512_DIGEST_LENGTH, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // 输出文件的校验码（即最后一个块的校验码）
        for (int i = 0; i < SHA512_DIGEST_LENGTH; i++) {
            printf("%02x", checksums[(num_blocks - 1) * SHA512_DIGEST_LENGTH + i]);
        }
        printf("\n");

        delete[] file_data;
        delete[] checksums;
    } else {
        // 子进程接收文件大小和数据
        std::size_t file_size;
        MPI_Bcast(&file_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

        uint8_t *file_data = new uint8_t[file_size];
        MPI_Bcast(file_data, file_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        // 计算子进程分配的部分数据的校验和
        uint8_t *obuf = new uint8_t[SHA512_DIGEST_LENGTH];
        calculate_checksum(file_data, file_size, obuf, rank, num_procs);

        // 将校验和结果发送给主进程
        MPI_Send(obuf, SHA512_DIGEST_LENGTH, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);

        delete[] file_data;
        delete[] obuf;
    }

    MPI_Finalize();
    return 0;
}
