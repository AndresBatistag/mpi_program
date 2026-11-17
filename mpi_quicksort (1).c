/*
 * ECE 525 - Lab 4, Exercise 2
 * Parallel QuickSort using MPICH
 *
 * The controller reads integers from ANY text file containing whitespace-separated
 * integers (one per line, or separated by spaces/tabs). Give the filename on
 * the command line; no filename is hardcoded in this program.
 * MPI_Scatter distributes the numbers, each process sorts its part,
 * MPI_Gather brings the sorted parts back, and the controller merges them.
 *
 * Compile: mpicc -O2 -Wall -Wextra mpi_quicksort.c -o mpi_quicksort
 * Pi cluster: mpiexec -n 8 -hostfile mpihost ./mpi_quicksort in3.txt
 * Other file: mpiexec -n 8 -hostfile mpihost ./mpi_quicksort any_list.txt
 * Local test: mpiexec -n 2 ./mpi_quicksort numbers.txt
 * Limit input: mpiexec -n 8 -hostfile mpihost ./mpi_quicksort in3.txt 1000000
 *
 * Only the controller needs the input text file. Every node needs the executable
 * in the same location (unless your cluster uses shared storage).
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <errno.h>

typedef struct {
    int value;
    int chunk;
    int index;
} HeapItem;

/* Load integers from the file. A limit of 0 means read the entire file. */
static int read_numbers(const char *filename, int limit, int **out)
{
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror(filename);
        return -1;
    }

    int capacity = (limit > 0 && limit < 65536) ? limit : 65536;
    int *numbers = malloc((size_t)capacity * sizeof(*numbers));
    if (!numbers) {
        fprintf(stderr, "Not enough memory to read the input file.\n");
        fclose(file);
        return -1;
    }

    int count = 0;
    int value;
    int result = 0;
    while ((limit == 0 || count < limit) &&
           (result = fscanf(file, "%d", &value)) == 1) {
        if (count == capacity) {
            if (capacity > INT_MAX / 2) {
                fprintf(stderr, "Input file has too many integers.\n");
                free(numbers);
                fclose(file);
                return -1;
            }
            capacity *= 2;
            int *bigger = realloc(numbers, (size_t)capacity * sizeof(*numbers));
            if (!bigger) {
                fprintf(stderr, "Not enough memory to read the input file.\n");
                free(numbers);
                fclose(file);
                return -1;
            }
            numbers = bigger;
        }
        numbers[count++] = value;
    }

    if (result == 0 && (limit == 0 || count < limit)) {
        fprintf(stderr, "Invalid integer in %s.\n", filename);
        free(numbers);
        fclose(file);
        return -1;
    }
    if (ferror(file)) {
        perror("Error reading input file");
        free(numbers);
        fclose(file);
        return -1;
    }
    fclose(file);

    if (count == 0) {
        fprintf(stderr, "The input file is empty.\n");
        free(numbers);
        return -1;
    }
    *out = numbers;
    return count;
}

/* In-place QuickSort, using recursion only for the smaller partition. */
static void quicksort(int *a, int left, int right)
{
    while (left < right) {
        int i = left;
        int j = right;
        int pivot = a[left + (right - left) / 2];

        while (i <= j) {
            while (a[i] < pivot) i++;
            while (a[j] > pivot) j--;

            if (i <= j) {
                int temp = a[i];
                a[i] = a[j];
                a[j] = temp;
                i++;
                j--;
            }
        }

        if (j - left < right - i) {
            if (left < j) quicksort(a, left, j);
            left = i;
        } else {
            if (i < right) quicksort(a, i, right);
            right = j;
        }
    }
}

static void heap_push(HeapItem *heap, int *size, HeapItem item)
{
    int i = (*size)++;
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (heap[parent].value <= item.value) break;
        heap[i] = heap[parent];
        i = parent;
    }
    heap[i] = item;
}

static HeapItem heap_pop(HeapItem *heap, int *size)
{
    HeapItem smallest = heap[0];
    HeapItem last = heap[--(*size)];
    int i = 0;

    while (2 * i + 1 < *size) {
        int child = 2 * i + 1;
        if (child + 1 < *size && heap[child + 1].value < heap[child].value)
            child++;
        if (last.value <= heap[child].value) break;
        heap[i] = heap[child];
        i = child;
    }
    if (*size > 0) heap[i] = last;
    return smallest;
}

/* Merge the sorted chunks using a min-heap (O(N log P)). */
static void merge_chunks(const int *chunks, int *merged, int chunk_size,
                         int processes)
{
    if (processes == 1) {
        memcpy(merged, chunks, (size_t)chunk_size * sizeof(*merged));
        return;
    }

    HeapItem *heap = malloc((size_t)processes * sizeof(*heap));
    if (!heap) {
        fprintf(stderr, "Could not allocate merge heap.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int heap_size = 0;
    for (int p = 0; p < processes; p++) {
        HeapItem item = {chunks[p * chunk_size], p, 0};
        heap_push(heap, &heap_size, item);
    }

    int total = chunk_size * processes;
    for (int i = 0; i < total; i++) {
        HeapItem item = heap_pop(heap, &heap_size);
        merged[i] = item.value;

        int next = item.index + 1;
        if (next < chunk_size) {
            HeapItem next_item = {
                chunks[item.chunk * chunk_size + next], item.chunk, next
            };
            heap_push(heap, &heap_size, next_item);
        }
    }
    free(heap);
}

int main(int argc, char **argv)
{
    int rank, processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    if (argc < 2 || argc > 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <integer_file.txt> [max_integers]\n", argv[0]);
            fprintf(stderr, "The file may contain integers separated by spaces or newlines.\n");
            fprintf(stderr, "Example: %s numbers.txt 100000\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int limit = 0;
    if (argc == 3) {
        char *end;
        errno = 0;
        long parsed = strtol(argv[2], &end, 10);
        if (errno != 0 || end == argv[2] || *end != '\0' ||
            parsed <= 0 || parsed > INT_MAX) {
            if (rank == 0)
                fprintf(stderr, "max_integers must be a positive integer.\n");
            MPI_Finalize();
            return 1;
        }
        limit = (int)parsed;
    }

    int count = 0;
    int *data = NULL;
    unsigned long long initial_sum = 0;

    /* Only rank 0 reads the file. File I/O is not included in the MPI timing. */
    if (rank == 0) {
        count = read_numbers(argv[1], limit, &data);
        if (count < 0) MPI_Abort(MPI_COMM_WORLD, 1);

        for (int i = 0; i < count; i++)
            initial_sum += (unsigned int)data[i];
    }

    MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Pad to a multiple of the process count to use MPI_Scatter/Gather. */
    long long chunk_ll = ((long long)count + processes - 1) / processes;
    long long padded_ll = chunk_ll * processes;
    if (padded_ll > INT_MAX) {
        if (rank == 0) fprintf(stderr, "Too many integers for this program.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int chunk_size = (int)chunk_ll;
    int padded_count = (int)padded_ll;
    int *local = malloc((size_t)chunk_size * sizeof(*local));
    int *merged = NULL;

    if (rank == 0) {
        int *resized = realloc(data, (size_t)padded_count * sizeof(*data));
        if (!resized) {
            fprintf(stderr, "Not enough memory for input array.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        data = resized;
        for (int i = count; i < padded_count; i++)
            data[i] = INT_MAX;
        merged = malloc((size_t)padded_count * sizeof(*merged));
    }

    if (!local || (rank == 0 && !merged)) {
        fprintf(stderr, "Rank %d: not enough memory.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Scatter(data, chunk_size, MPI_INT,
                local, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    quicksort(local, 0, chunk_size - 1);

    MPI_Gather(local, chunk_size, MPI_INT,
               data, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    double elapsed = 0.0;
    if (rank == 0) {
        merge_chunks(data, merged, chunk_size, processes);
        elapsed = MPI_Wtime() - start;
    }

    /* Gather processor names so the output shows which nodes did the work. */
    char hostname[MPI_MAX_PROCESSOR_NAME] = {0};
    int name_length;
    MPI_Get_processor_name(hostname, &name_length);
    char *hostnames = NULL;
    if (rank == 0) {
        hostnames = calloc((size_t)processes, MPI_MAX_PROCESSOR_NAME);
        if (!hostnames) MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Gather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               hostnames, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        int sorted = 1;
        unsigned long long final_sum = 0;
        for (int i = 0; i < count; i++) {
            if (i > 0 && merged[i - 1] > merged[i]) sorted = 0;
            final_sum += (unsigned int)merged[i];
        }

        printf("Input file: %s\n", argv[1]);
        printf("Integers sorted: %d\n", count);
        printf("MPI processes: %d\n", processes);
        printf("Integers per process: %d\n", chunk_size);
        printf("MPI time (scatter + sort + gather + merge): %.6f seconds\n", elapsed);
        printf("Sorted and checksum verified: %s\n",
               (sorted && initial_sum == final_sum) ? "YES" : "NO");
        printf("Processes ran on:\n");
        for (int p = 0; p < processes; p++)
            printf("  Rank %d: %s\n", p, hostnames + (size_t)p * MPI_MAX_PROCESSOR_NAME);
    }

    free(local);
    free(data);
    free(merged);
    free(hostnames);
    MPI_Finalize();
    return 0;
}
