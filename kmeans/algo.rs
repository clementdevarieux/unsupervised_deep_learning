fn k_means(
    inputs_train: &[f32],
    number_of_clusters: usize,
    dimensions_of_inputs: usize,
    number_of_points: usize,
) -> Vec<Vec<f32>> {
    // initialiser des centres randoms
    let mut vec_of_mu_k: Vec<Vec<f32>> = Vec::with_capacity(number_of_clusters);
    // vec_of_mu_k= [mu_0, mu_1 ..., mu_k]
    // mu_x = le centre du cluster x
    // mu_x a le meme nombre d'éléments que une image
    for k in 0..number_of_clusters{
        let mut mu_k : Vec<f32> = Vec::with_capacity(dimensions_of_inputs);
        for j in 0..dimensions_of_inputs{
            mu_k.push(rand::thread_rng().gen_range(-1f32..1f32));
        }
        vec_of_mu_k.push(mu_k);
    }

    // création ensemble Sk
    // Pour chaque points on vérifie à quel centre il appartient
    // X -> pour tout point -> on vérifie s'il est plus proche d'un centre ou d'un autre
    let mut old_vec_of_mu_k : Vec<Vec<f32>> = Vec::with_capacity(number_of_clusters);
    for k in 0..number_of_clusters{
        let temp_vec: Vec<f32> = vec![0.0; dimensions_of_inputs];
        old_vec_of_mu_k.push(temp_vec);
    }
    let mut count = 0;
    while old_vec_of_mu_k != vec_of_mu_k && count <= 100 {
        let mut vec_of_Sk: Vec<Vec<Vec<f32>>> = Vec::with_capacity(number_of_clusters);

        for k in 0..number_of_clusters {
            let mut S_k: Vec<Vec<f32>> = Vec::new();
            for n in 0..number_of_points {
                let mut distance_k: f32 = 0.0;
                for j in 0..dimensions_of_inputs {
                    distance_k += (inputs_train[n * dimensions_of_inputs + j] - vec_of_mu_k[k][j])*(inputs_train[n * dimensions_of_inputs + j] - vec_of_mu_k[k][j]);
                }
                distance_k = distance_k.sqrt();
                for l in 0..number_of_clusters {
                    if l != k {
                        let mut distance_l: f32 = 0.0;
                        for j in 0..dimensions_of_inputs {
                            distance_l += (inputs_train[n * dimensions_of_inputs + j] - vec_of_mu_k[l][j])*(inputs_train[n * dimensions_of_inputs + j] - vec_of_mu_k[l][j]);
                        }
                        distance_l = distance_l.sqrt();
                        if distance_k <= distance_l {
                            let mut vec_to_push = Vec::with_capacity(dimensions_of_inputs);
                            for i in 0..dimensions_of_inputs {
                                vec_to_push.push(inputs_train[n * dimensions_of_inputs + i]);
                            }
                            S_k.push(vec_to_push);
                        }
                    }
                }
            }
            vec_of_Sk.push(S_k);
        }

        //update mu_k
        old_vec_of_mu_k = vec_of_mu_k;
        vec_of_mu_k = Vec::with_capacity(number_of_clusters);
        for k in 0..number_of_clusters {
            let mut mu_k: Vec<f32> = vec![0.0; dimensions_of_inputs];
            for n in &vec_of_Sk[k] {
                for i in 0..dimensions_of_inputs {
                    mu_k[i] += n[i] / vec_of_Sk[k].len() as f32;
                }
            }
            vec_of_mu_k.push(mu_k);
        }
        count += 1;
    }
    vec_of_mu_k

}
