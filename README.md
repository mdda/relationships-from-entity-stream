# relationships-from-entity-stream

Research idea, extending the work in the DeepMind "Relation Networks" paper

### Abstract

    Relational reasoning is a central component of intelligent behavior, 
    but has proven difficult for neural networks to learn.  The Relation Network (RN) 
    module was recently proposed by DeepMind to solve such problems, 
    and demonstrated state-of-the-art results on a number of datasets.  However, 
    the RN module scales quadratically in the size of the input, 
    since it calculates relationship factors between every patch in the visual field, 
    including those that do not correspond to entities.  In this paper, 
    we describe an architecture that enables relationships to be determined 
    from a stream of entities obtained by an attention mechanism over the input field.  The model 
    is trained end-to-end, and demonstrates 
    equivalent performance with greater interpretability 
    while requiring only a fraction of the model parameters of the original RN module.  


