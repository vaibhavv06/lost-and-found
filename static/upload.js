let storageRef = firebase.storage().ref('/')

upload_face_image = () => {
  let data = new FormData()
  let file = document.getElementById("face-input").files[0]
  data.set('face',file)
  axios.post('http://localhost:4000/face',data,{
    headers:  {
      'content-type': 'multipart/form-data'
    }
  }).then(res => {
    console.log(res.data)
    let index = res.data.Index
    let address = index+".jpg"
    storageRef.child(address).getDownloadURL()
    .then(url => console.log("URL: "+url))
  })
  .catch(err => console.log(err))
}

upload_fingerprint = () => {
  let formData = new FormData()
  let file = document.getElementById("fingerprint-input").files[0]
  let sel = document.getElementById("finger-type")
  let text = sel.options[sel.selectedIndex].value
  formData.append('fingerprint',file)
  formData.append('text',text)
  axios.post('http://localhost:4000/fingerprint',formData,{
    headers: {
      'content-type': 'multipart/form-data'
    }
  }).then(res => {
    console.log(res.data)
    let index = res.data.Index
    let address = index+".jpg"
    storageRef.child(address).getDownloadURL()
    .then(url => console.log("URL: "+url))
  })
  .catch(err => console.log(err))
}