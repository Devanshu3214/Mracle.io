$(".custom-file-input").on("change", function() {
    let fileName = $(this).val().split("\\").pop();
    $(this).siblings(".custom-file-label").addClass("selected").html(fileName);
});


document.getElementById('uploadForm').onsubmit = async function (e) {
    e.preventDefault();

    document.getElementById('loader').style.display = 'block';

    const formData = new FormData(this);
    const response = await fetch('/describe', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    document.getElementById('loader').style.display = 'none';
    document.getElementById('output').innerHTML = `<pre>${JSON.stringify(result, null, 2)}</pre>`;
};