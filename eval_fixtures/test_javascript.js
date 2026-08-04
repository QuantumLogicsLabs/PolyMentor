function processData(data) {
    // Bug: assignment in condition (if data = null) instead of (if data === null)
    if (data = null) {
        return false;
    }
    
    // Bug: var usage (anti-pattern)
    var result = [];
    
    // Bug: Promise without catch
    fetch('https://api.example.com/data')
        .then(function(res) {
            return res.json();
        })
        .then(function(json) {
            result.push(json);
        });
        
    return result;
}
