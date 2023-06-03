const uploadInput = document.getElementById("image-upload");
const uploadedImage = document.getElementById("uploaded-image");
const hidden = document.getElementById("is-hidden");
const hoverCheck = document.getElementById("hover-check");
uploadInput.addEventListener("change", function (event) {
  const file = event.target.files[0];
  const reader = new FileReader();

  reader.onload = function (e) {
    uploadedImage.src = e.target.result;
    uploadedImage.style.display = "inline-block";
    hidden.style.display = "none";
    hoverCheck.addEventListener("mouseover", function () {
      uploadedImage.style.display = "none";
      hidden.style.display = "block";
    });

    hoverCheck.addEventListener("mouseout", function () {
      uploadedImage.style.display = "block";
      hidden.style.display = "none";
    });
  };

  reader.readAsDataURL(file);
});

function sendImage() {
  var input = document.getElementById("image-upload");
  var file = input.files[0];
  var formData = new FormData();
  formData.append("image", file);

  $.ajax({
    type: "POST",
    url: "/upload",
    data: formData,
    processData: false,
    contentType: false,
    success: function (response) {
      var listImages = response.split(",");
      console.log(listImages)
      var renderHtml = "";
      for (var i = 0; i < 5; i++) {
        renderHtml= renderHtml + `
          <div class="photo-results">
            <img src="../static/train/${listImages[i].trim()}" alt="" class="photo-results-img" />
          </div>
        `;
      }
      document.querySelector(".photo-first").innerHTML = renderHtml
      renderHtml = "";
      for (var i = 5; i < 10; i++) {
        renderHtml= renderHtml + `
          <div class="photo-results">
            <img src="../static/train/${listImages[i].trim()}" alt="" class="photo-results-img" />
          </div>
        `;
      }
      document.querySelector(".photo-second").innerHTML = renderHtml
    },
    error: function (error) {
      alert("Lỗi khi gửi ảnh: " + error);
    },
  });
}
