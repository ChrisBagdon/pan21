# pan21

PAN Shared Task 2021 Hate Speech Spreaders on Twitter

Working notes and Latex files can be found in the Paper directory.

System was built under a very short deadline and is not optimally organized

Usage:

1. Create Logistic Regression model: 
  Set Hyper Params in build_reg_model.py
  
    <code> python build_reg_model.py "PATH/TO/pan21-author-profiling-training-2021-03-14" </code>
  
2. Create RoBERTa Model:
  Set Hyper Params in build_a_bert.py
  
    <code> python build_a_bert.py "PATH/TO/pan21-author-profiling-training-2021-03-14" </code>
    
3. Create Meta-Classifier:
  Set Hyper Params in build_combo_model.py
  
    <code> python build_combo_model.py "PATH/TO/pan21-author-profiling-training-2021-03-14" </code>
    
4. Run on test set and create prediction XMLs:

    <code> make_predictions.py "PATH/TO/test_set_XML_dir" "PATH/TO/desired_ouput_dir"</code>
    
    
Cross_val.py can be used on its own to perform 10 fold crossfold validation on the training set. Simply edit Hyper Params and run



