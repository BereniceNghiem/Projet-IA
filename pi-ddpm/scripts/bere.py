import numpy as np

def inspect_npz_file(file_path):
    """
    Affiche les tableaux contenus dans un fichier .npz et leurs shapes.
    
    Args:
        file_path (str): Chemin vers le fichier .npz.
    """
    try:
        # Charger le fichier .npz
        data = np.load(file_path)
        
        # Afficher les noms des tableaux (colonnes) disponibles
        print("\n🔍 Contenu du fichier NPZ :")
        print("-----------------------------")
        print("Tableaux (colonnes) disponibles:", list(data.keys()))  # ou data.files
        
        # Afficher le shape de chaque tableau
        print("\n📏 Détails des tableaux :")
        print("-----------------------------")
        for key in data.keys():
            array = data[key]
            print(f"- {key}:")
            print(f"  Shape: {array.shape}")
            print(f"  Type: {array.dtype}")
            print(f"  Exemple de valeurs (5 premières): {array[:5] if array.size > 0 else '[]'}\n")
        
    except FileNotFoundError:
        print(f"❌ Erreur: Fichier non trouvé à l'emplacement {file_path}")
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du fichier: {e}")

# Exemple d'utilisation
file_path = "/home/ids/bnghiem-23/Projet-IA-Telecom-Paris/pi-ddpm/data/w2s_test.npz"  # Remplacez par votre chemin
inspect_npz_file(file_path)