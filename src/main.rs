use std::collections::HashMap;
use std::fs;
use walkdir::WalkDir;

const THRESHOLD: i32 = 100;

fn total_count(bow: &HashMap<String, i32>) -> i32 {
    let mut count = 0;
    for (_, frequency) in bow {
        if *frequency < THRESHOLD {
            continue;
        }
        count += frequency;
    }
    count
}

fn add_dir_to_bow(path: String, bow: &mut HashMap<String, i32>) -> Result<(), std::io::Error> {
    for file in WalkDir::new(path).into_iter().filter_map(|f| f.ok()) {
        if file.metadata()?.is_file() {
            let filename = file.path().to_str().unwrap();
            add_file_to_bow(filename, bow)?;
        }
    }
    Ok(())
}

fn add_file_to_bow(filepath: &str, bow: &mut HashMap<String, i32>) -> Result<(), std::io::Error> {
    let bytes = fs::read(filepath)?;
    let contents = String::from_utf8_lossy(&bytes);
    let tokens: Vec<&str> = contents.split_whitespace().collect();
    for &each_token in tokens.iter() {
        bow.entry(each_token.to_uppercase())
            .and_modify(|count| *count += 1)
            .or_insert(1);
    }
    Ok(())
}

fn classify_file(
    ham_bow: &HashMap<String, i32>,
    ham_total_count: i32,
    spam_bow: &HashMap<String, i32>,
    spam_total_count: i32,
    file_path: &str,
) -> Result<(f64, f64), std::io::Error> {
    let mut file_bow = HashMap::new();
    add_file_to_bow(file_path, &mut file_bow)?;
    let mut dp = 0.0;
    let mut spam_dp = 0.0;
    let mut ham_dp = 0.0;
    let total_count = ham_total_count + spam_total_count;
    let ham_p = (ham_total_count as f64 / total_count as f64).ln();
    let spam_p = (spam_total_count as f64 / total_count as f64).ln();
    for (token, _) in file_bow {
        let spam_freq = *(spam_bow.get(&token).unwrap_or(&0));
        let ham_freq = *(ham_bow.get(&token).unwrap_or(&0));

        let n = spam_freq + ham_freq;
        if n < THRESHOLD {
            continue;
        }
        if spam_freq != 0 {
            spam_dp += (spam_freq as f64 / spam_total_count as f64).ln();
        }
        if ham_freq != 0 {
            ham_dp += (ham_freq as f64 / ham_total_count as f64).ln();
        }

        if n != 0 {
            dp += (n as f64 / total_count as f64).ln();
        }
    }
    Ok((spam_dp + spam_p - dp, ham_dp + ham_p - dp))
}

fn classify_dir(
    ham_bow: &HashMap<String, i32>,
    ham_total_count: i32,
    spam_bow: &HashMap<String, i32>,
    spam_total_count: i32,
    dir_path: &str,
) -> Result<(i32, i32), std::io::Error> {
    let mut ham_outcome_count = 0;
    let mut spam_outcome_count = 0;
    for file in WalkDir::new(dir_path).into_iter().filter_map(|f| f.ok()) {
        if !file.metadata()?.is_file() {
            continue;
        }
        let file_path = file.path().to_str().unwrap();
        let (spam_p, ham_p) = classify_file(
            &ham_bow,
            ham_total_count,
            &spam_bow,
            spam_total_count,
            file_path,
        )?;
        if spam_p > ham_p {
            spam_outcome_count += 1;
        } else {
            ham_outcome_count += 1;
        }
    }
    Ok((spam_outcome_count, ham_outcome_count))
}

fn main() -> Result<(), std::io::Error> {
    println!("Training..");
    let mut ham_bow: HashMap<String, i32> = HashMap::new();
    let mut spam_bow: HashMap<String, i32> = HashMap::new();

    for i in 1..=5 {
        add_dir_to_bow(format!("data/enron{i}/ham"), &mut ham_bow)?;
        add_dir_to_bow(format!("data/enron{i}/spam"), &mut spam_bow)?;
    }

    let ham_total_count = total_count(&ham_bow);
    let spam_total_count = total_count(&spam_bow);

    println!("Classifying ham..");
    let (spam_outcome_count, ham_outcome_count) = classify_dir(
        &ham_bow,
        ham_total_count,
        &spam_bow,
        spam_total_count,
        "data/enron6/ham",
    )?;

    println!("Spam count : {spam_outcome_count}");
    println!("Ham count : {ham_outcome_count}");

    println!("Classifying spam..");
    let (spam_outcome_count, ham_outcome_count) = classify_dir(
        &ham_bow,
        ham_total_count,
        &spam_bow,
        spam_total_count,
        "data/enron6/spam",
    )?;

    println!("Spam count : {spam_outcome_count}");
    println!("Ham count : {ham_outcome_count}");

    Ok(())
}
