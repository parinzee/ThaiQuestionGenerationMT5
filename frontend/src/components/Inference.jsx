import axios from "axios";
import { useState } from "react";
import { useForm } from "react-hook-form";
import Settings from "./Settings";
import ModelInput from "./ModelInput";

function Inference() {
  const { register, handleSubmit } = useForm();
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState("");

  const onSubmit = (data) => {
    setLoading(true);
    axios
      .post("http://ec2-54-169-5-52.ap-southeast-1.compute.amazonaws.com/", {
        input_text: data.input_text,
        model: data.model,
        num_beams: parseInt(data.num_beams),
        max_length: 2048,
        repetition_penalty: parseFloat(data.repetition_penalty),
        early_stopping: true,
        top_p: parseInt(data.top_p),
        top_k: parseInt(data.top_k),
        num_return_sequences: 1,
      })
      .then((response) => {
        setData(response.data);
        setLoading(false);
      });
  };

  return (
    <form
      onSubmit={handleSubmit(onSubmit)}
      className="flex flex-col justify-between row-span-6 md:col-span-2 h-full"
    >
      <ModelInput register={register} loading={loading} data={data} />
      <Settings register={register} />
    </form>
  );
}

export default Inference;
