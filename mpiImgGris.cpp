#include <iostream>
#include <vector>
#include <mpi.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static int rgb=3;

std::vector<uint8_t> recorrerImgEscalaGris(const uint8_t * img, int width, int height){
    std::vector<uint8_t> buffer(width * height);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = (i * width + j) * rgb;
            int gray_scale = 0.21*img[index]+0.72*img[index+1]+0.07*img[index+2];
            int index2 = (i * width + j);
            buffer[index2] = gray_scale;
        }
    }
    return buffer;
}

int main(int argc, char** argv) {
    MPI_Init(&argc,&argv);
    int rank,nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int width, height, channels;
    int filas_por_ranks;
    int filas_r;
    uint8_t* rgb_pixels;
    std::vector<uint8_t > imgFinal;

    if(rank==0){
        rgb_pixels =
                stbi_load("../image01.jpg", &width, &height, &channels, STBI_rgb);
        filas_por_ranks=std::ceil((double )height/nprocs);
        imgFinal.resize(width*height);
    }
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&filas_por_ranks, 1,MPI_INT,0,MPI_COMM_WORLD);

    int inicio_filas=rank*filas_por_ranks;
    int fin_filas = inicio_filas+filas_por_ranks;

    if(rank==nprocs-1){
        fin_filas=height;
    }
    filas_r=fin_filas-inicio_filas;
    int pixel_por_filas = width*rgb;
    int pixel_por_filas_reto = width;

    std::vector<uint8_t> buffer((fin_filas-inicio_filas)*pixel_por_filas);

    MPI_Scatter(rgb_pixels, pixel_por_filas * filas_r,MPI_UNSIGNED_CHAR,
                buffer.data(), pixel_por_filas * filas_r,MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    std::vector<uint8_t> gris= recorrerImgEscalaGris(buffer.data(),width,filas_r);

    MPI_Gather(gris.data(), pixel_por_filas_reto*filas_r,MPI_UNSIGNED_CHAR,
               imgFinal.data(), pixel_por_filas_reto*filas_r,MPI_UNSIGNED_CHAR,
               0,MPI_COMM_WORLD);

    if(rank==0){
        uint8_t* gray_pixels = imgFinal.data();
        stbi_write_png( "../img-grfis.png", width, height,
                        STBI_grey, gray_pixels, width );
    }

    MPI_Finalize();

    return 0;
}
