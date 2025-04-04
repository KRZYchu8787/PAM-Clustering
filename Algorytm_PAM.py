import numpy as np

def cluster_PAM(X, k):
    """
    Partitioning Around Medoids (PAM) Algorithm
    
    Parameters:
    X : np.ndarray
        Dane wejściowe w postaci macierzy NumPy (punkty, wymiar).
    k : int
        Liczba medoidów (klastrów).
    
    Returns:
    medoids : list
        Indeksy wybranych medoidów.
    clusters : list
        Lista przypisań punktów do klastrów.
    final_cost : int
        Koszt znalezionego grupowania
    """

    # Obsługa wyjątków
    try:
        if not isinstance(X, np.ndarray):
            raise ValueError("Input data X must be a NumPy array.")
        if len(X.shape) != 2:
            raise ValueError("Input data X must be a 2D NumPy array (n_samples, n_features).")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Number of clusters k must be a positive integer.")
        if k > X.shape[0]:
            raise ValueError("Number of clusters k cannot exceed the number of data points.")
        if np.any(np.isnan(X)):
            raise ValueError("Input data X contains NaN values. Please handle missing data before running the algorithm.")

        # Liczymy wielką macierz kosztu, co pozwoli nie liczyć później wiele razy tego samego
        cost_matrix = np.array([np.linalg.norm(X - X[i], axis=1) for i in range(X.shape[0])]) # odległość każdego punktu od każdego punktu

        # Inicjalizacja - losowe wybranie k medoidów
        n_samples = X.shape[0]
        medoids = list(np.random.choice(n_samples, k, replace=False))  # Indeksy początkowych medoidów

        # Liczymy pierwszy koszt z wylosowanych medoidów
        sub_matrix = cost_matrix[:, medoids]
        best_cost = np.sum(np.min(sub_matrix, axis=1)) # bierzemy wartości minimalne z podmacierzy kosztu
        
        while True:

            medoids_before_this_iteration = medoids.copy()

            for medoid in medoids_before_this_iteration:
                
                best_medoid = medoid  
             
                # Testowanie zamian punktów na medoidy
                for candidate in range(n_samples):

                    if candidate not in medoids:  # Tylko punkty niebędące aktualnie medoidami mogą zastąpić stary medoid
                        test_medoids = medoids.copy()
                        test_medoids.remove(medoid)
                        test_medoids.append(candidate)
                        
                        # Liczymy czy ta zamiana daje lepszy koszt
                        test_matrix = cost_matrix[:, test_medoids]
                        test_cost = np.sum(np.min(test_matrix, axis=1)) 
                        
                        # Sprawdzamy czy ten kandydat jest dotychczas najlepszy
                        if test_cost < best_cost:
                            best_cost = test_cost
                            best_medoid = candidate
                
                # Po przetestowaniu wszystkich kandydatów zamieniamy rozważany medoid na najlepszego zamiennika
                medoids.remove(medoid)
                medoids.append(best_medoid)
                 
            # Jeśli po przejściu przez wszystkie medoidy nic się nie zmieniło to koniec
            if set(medoids) == set(medoids_before_this_iteration):
                break

        # Definitywne przypisanie punktów do klastrów, ostateczny koszt, finalne klastry
        final_matrix = cost_matrix[:, medoids]
        final_cost = np.sum(np.min(final_matrix, axis=1))
        clusters = np.argmin(final_matrix, axis=1) # wektor kolumnowy wskazujący, do którego medoidu należy punkt

        return medoids, clusters, final_cost

    except Exception as e:
        print(f"An error occurred: {e}")
        raise