def inspect_model(model, name="Model"):
    """Displays key information about the model."""
    print(f"\n{'='*20}")
    print(f"Inspecting: {name}")
    print(f"{'='*20}")
    
    print(f"Type: {type(model)}")
    
    # Check for common sklearn attributes
    if hasattr(model, 'get_params'):
        print("\nParameters:")
        params = model.get_params()
        for param, value in params.items():
            print(f"  {param}: {value}")
            
    if hasattr(model, 'classes_'):
        print(f"\nClasses: {model.classes_}")
        
    if hasattr(model, 'feature_names_in_'):
        print(f"\nFeatureNames: {model.feature_names_in_}")
        
    print(f"{'='*20}\n")
