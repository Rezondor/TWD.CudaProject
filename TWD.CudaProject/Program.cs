using ManagedCuda;
using ManagedCuda.VectorTypes;

namespace TWD.CudaProject;

public class Program
{
    static void Main(string[] args)
    {
        int N = 1_048_576;
        float[] hostA = new float[N];
        float[] hostB = new float[N];
        float[] hostC = new float[N];

        for (int i = 0; i < N; i++)
        {
            hostA[i] = i;
            hostB[i] = 2 * i;
        }

        // Инициализация CUDA
        var context = new CudaContext();

        // Загрузка PTX и ядра
        var module = context.LoadModule("Cores/vector_add.ptx");
        var kernel = new CudaKernel("VectorAdd", module, context);

        // Выделение памяти на GPU
        var devA = new CudaDeviceVariable<float>(N);
        var devB = new CudaDeviceVariable<float>(N);
        var devC = new CudaDeviceVariable<float>(N);

        // Копирование данных на GPU
        devA.CopyToDevice(hostA);
        devB.CopyToDevice(hostB);

        // Настройка параметров запуска ядра
        int threadsPerBlock = 512;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        kernel.BlockDimensions = new dim3(threadsPerBlock);
        kernel.GridDimensions = new dim3(blocksPerGrid);

        // Вызов ядра
        kernel.Run(devA.DevicePointer, devB.DevicePointer, devC.DevicePointer, N);

        // Копирование результата обратно
        devC.CopyToHost(hostC);

        // Проверка результата
        for (int i = 0; i < 100; i++)
        {
            Console.WriteLine($"{hostA[i]} + {hostB[i]} = {hostC[i]}");
        }

        // Очистка
        devA.Dispose();
        devB.Dispose();
        devC.Dispose();
        context.Dispose();
    }
}
