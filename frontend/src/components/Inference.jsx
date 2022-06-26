import axios from "axios";
import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import Settings from "./Settings";
import ModelInput from "./ModelInput";

function Inference() {
  const { register, watch, handleSubmit, setValue } = useForm();
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState("");

  const onSubmit = (data) => {
    setLoading(true);
    axios
      .post("https://ec2-54-169-5-52.ap-southeast-1.compute.amazonaws.com/", {
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

  useEffect(() => {
    const subscription = watch((value, { name, type }) => {
      if (name === "model" && type === "change") {
        if (value.model === "default" || value.model === "number_separated") {
          setValue("num_beams", 3)
          setValue("repetition_penalty", 2.75)
        } else if (value.model === "augmented_number_separated") {
          setValue("num_beams", 4)
          setValue("repetition_penalty", 3.1)
        }
      }
    });
    return () => subscription.unsubscribe();

  }, [watch])

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
