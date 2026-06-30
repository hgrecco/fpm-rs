use fpm_rs::Array2;
use serde_json::json;

#[test]
fn array_json_round_trip_preserves_shape_and_values() {
    let array = Array2::from_vec((2, 3), vec![1_u16, 2, 3, 4, 5, 6]).unwrap();
    let encoded = serde_json::to_string(&array).unwrap();
    let decoded: Array2<u16> = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, array);
}

#[test]
fn array_deserialization_rejects_invalid_private_invariants() {
    for invalid in [
        json!({"height": 2, "width": 2, "data": [1.0, 2.0, 3.0]}),
        json!({"height": 0, "width": 2, "data": []}),
        json!({"height": usize::MAX, "width": 2, "data": []}),
    ] {
        assert!(serde_json::from_value::<Array2<f64>>(invalid).is_err());
    }
}
